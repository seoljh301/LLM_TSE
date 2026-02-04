import torch
import torch.nn as nn
import torch.nn.functional as F

class DualProjectionBlock(nn.Module):
    """
    Corresponds to 'Dual Projection Fusion' in VORTEX paper.
    (Mapped to ConnectBlock in VORTEX code)
    Simple late fusion: Projects both inputs independently and sums them.
    Good for low-overlap scenarios.
    """
    def __init__(self, d_model):
        super().__init__()
        self.res_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.cross_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, residual, cross_output):
        # residual: e_spk (B, D)
        # cross_output: e_txt (B, D)
        out = self.res_proj(residual) + self.cross_proj(cross_output)
        return self.norm(out + residual)

class AdaptiveFusionBlock(nn.Module):
    """
    Corresponds to 'Adaptive Fusion' in VORTEX paper.
    (Mapped to LightHAMBlock in VORTEX code)
    Early fusion with gating: Computes weights for two signals via Softmax.
    """
    def __init__(self, d_model):
        super().__init__()
        self.scale_proj = nn.Linear(d_model * 2, d_model * 2)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, residual, cross_output):
        # Concatenate: (B, 2D)
        concat = torch.cat([cross_output, residual], dim=-1)
        
        # Scale projection -> (B, 2, D)
        scale = self.scale_proj(concat)
        scale = scale.view(residual.size(0), 2, -1) 
        
        # Softmax over the 2 branches
        weight = F.softmax(scale, dim=1) # (B, 2, D)
        
        # Weighted sum: w_cross * cross + w_res * residual
        fused = weight[:, 0, :] * cross_output + weight[:, 1, :] * residual
        
        out = self.proj(fused)
        return self.norm(out + residual)

class MultiScaleFusionBlock(nn.Module):
    """
    Corresponds to 'Multi-scale Fusion' in VORTEX paper.
    (Mapped to MultiScaleLightHAMBlock in VORTEX code)
    Uses multi-scale convolutions to capture context.
    Since LLM-TSE uses global vectors (T=1), we treat them as length-1 sequences with padding.
    """
    def __init__(self, d_model, kernel_sizes=[1, 3, 5]):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                # Input: (B, 2D, 1) -> Output: (B, D, 1)
                nn.Conv1d(d_model * 2, d_model, kernel_size=k, padding=k//2),
                nn.Sigmoid()
            ) for k in kernel_sizes
        ])
        self.scale_weight_proj = nn.Linear(d_model * 2, len(kernel_sizes))
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, residual, cross_output):
        # Inputs are (B, D). Convert to (B, 2D, 1) for Conv1d
        x_cat = torch.cat([cross_output, residual], dim=-1) # (B, 2D)
        x_in = x_cat.unsqueeze(-1) # (B, 2D, 1)

        # Scale-wise gating
        gated_outs = []
        for conv in self.scales:
            # gate: (B, D, 1)
            gate = conv(x_in).squeeze(-1) # (B, D)
            # Gated fusion
            out = gate * cross_output + (1 - gate) * residual
            gated_outs.append(out)

        # Calculate weights for each scale: (B, n_scales)
        scale_logits = self.scale_weight_proj(x_cat)
        scale_weights = F.softmax(scale_logits, dim=-1)

        # Weighted sum of scales
        fused = sum(scale_weights[:, i].unsqueeze(-1) * gated_outs[i] for i in range(len(gated_outs)))

        out = self.proj(fused)
        return self.norm(out + residual)

class DAMFusion(nn.Module):
    """
    Dynamic Allocation Multi-branch (DAM) Fusion.
    Integrates Dual Projection, Adaptive Fusion, and Multi-scale Fusion
    with a dynamic gating mechanism.
    """
    def __init__(self, d_model, dropout_p=0.1):
        super().__init__()
        
        # 1. Dual Projection (Low overlap specialist)
        self.low_branch = DualProjectionBlock(d_model)
        # 2. Adaptive Fusion (Mid overlap specialist)
        self.mid_branch = AdaptiveFusionBlock(d_model)
        # 3. Multi-scale Fusion (High overlap specialist)
        self.high_branch = MultiScaleFusionBlock(d_model)

        self.branches = [self.low_branch, self.mid_branch, self.high_branch]
        
        # Adapters for each branch (from VORTEX code)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.LayerNorm(d_model)
            ) for _ in range(3)
        ])

        # Dynamic Gating Network
        # Input: Concat(residual, cross) -> Weights for 3 branches
        self.gating = nn.Sequential(
            nn.Linear(d_model * 2, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, e_spk, e_txt):
        """
        e_spk: Enrollment embedding (Audio Cue) -> Acts as 'residual'
        e_txt: Text embedding (Text Cue) -> Acts as 'cross_output'
        """
        residual = e_spk
        cross_out = e_txt

        # Branch execution
        h_low = self.low_branch(residual, cross_out)
        h_mid = self.mid_branch(residual, cross_out)
        h_high = self.high_branch(residual, cross_out)

        # Adapter application
        h_low = self.adapters[0](h_low)
        h_mid = self.adapters[1](h_mid)
        h_high = self.adapters[2](h_high)

        # Dynamic Gating
        gate_input = torch.cat([residual, cross_out], dim=-1)
        weights = self.gating(gate_input) # (B, 3)

        w_low = weights[:, 0:1]
        w_mid = weights[:, 1:2]
        w_high = weights[:, 2:3]

        # Weighted Sum
        fused = (w_low * h_low) + (w_mid * h_mid) + (w_high * h_high)
        
        return fused

class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Fusion.
    Text embedding modulates the speaker embedding via affine transformation.
    gamma(e_txt) * e_spk + beta(e_txt)
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            # Simple linear projection to generate gamma and beta
        )
    
    def forward(self, e_spk, e_txt):
        # e_spk: (B, D)
        # e_txt: (B, D)
        params = self.net(e_txt)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma * e_spk + beta

class CueFusion(nn.Module):
    """
    Wrapper class compatible with LLM-TSE interface.
    Defaults to FiLM Fusion.
    """
    def __init__(self, dim: int, fusion_type="film"):
        super().__init__()
        if fusion_type == "dam":
            self.fusion = DAMFusion(dim)
        elif fusion_type == "film":
            self.fusion = FiLMFusion(dim)
        else:
            self.fusion = FiLMFusion(dim)
            
    def forward(self, e_spk, e_txt):
        return self.fusion(e_spk, e_txt)
