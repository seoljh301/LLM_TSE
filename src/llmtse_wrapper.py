import torch
import torch.nn as nn
from .text_encoder import LLMTextEncoder
from .fusion import CueFusion
from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape
from asteroid.models.base_models import _shape_reconstructed, _unsqueeze_to_3d

class LLMTSEWrapper(nn.Module):
    """
    Wraps SpeakerBeam to incorporate LLM-based text conditioning.
    Reflecting 'Typing to Listen at the Cocktail Party' architecture.
    """
    def __init__(self, base, spk_dim: int,
                 text_model_name: str = "meta-llama/Llama-2-7b-hf",
                 use_lora: bool = False,
                 load_in_4bit: bool = False,
                 allow_text_only: bool = True,
                 fusion_type: str = "concat"): # 'concat', 'film' or 'dam'
        super().__init__()
        self.base = base
        # LLM Text Encoder
        self.txt = LLMTextEncoder(text_model_name, out_dim=spk_dim, use_lora=use_lora, load_in_4bit=load_in_4bit)
        
        # Advanced Fusion Mechanism
        self.fuse = CueFusion(dim=spk_dim, fusion_type=fusion_type)
        
        self.allow_text_only = allow_text_only
        self.spk_dim = spk_dim
        self.text_model_name = text_model_name
        self.use_lora = use_lora
        self.fusion_type = fusion_type

    def forward(self, wav, enrollment=None, texts=None):
        if not (hasattr(self.base, "auxiliary") and hasattr(self.base, "masker")):
            raise RuntimeError("Base model must expose auxiliary and masker.")

        # Following logic from BaseEncoderMaskerDecoderInformed.forward
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.base.forward_encoder(wav)

        # Check for None or empty tensor
        is_enroll_present = enrollment is not None and enrollment.numel() > 0
        e_spk = self.base.auxiliary(enrollment) if is_enroll_present else None
        
        # Text input handling
        e_txt = None
        if texts is not None:
            # texts는 리스트 형태여야 함
            if isinstance(texts, str):
                texts = [texts]
            e_txt = self.txt(texts)
            # Debug print (Run once)
            if not hasattr(self, "_debug_printed"):
                print(f"[DEBUG] Text: {texts[0]}")
                print(f"[DEBUG] e_txt stats: Mean={e_txt.mean().item():.4f}, Std={e_txt.std().item():.4f}")
                self._debug_printed = True

        if e_spk is None and e_txt is None:
            raise ValueError("Need enrollment or texts")

        # Fusion Logic
        if e_spk is None:
            if not self.allow_text_only:
                raise ValueError("text-only disabled")
            # Text Only Mode: Use text embedding as the enrollment vector directly
            z = e_txt
        elif e_txt is None:
            # Audio Only Mode
            z = e_spk
        else:
            # Multi-modal Mode: Fusion
            z = self.fuse(e_spk, e_txt)
            
        if hasattr(self, "_debug_printed") and self._debug_printed is True:
             # Print z stats only once right after e_txt
             print(f"[DEBUG] z (Fused) stats: Mean={z.mean().item():.4f}, Std={z.std().item():.4f}")
             self._debug_printed = "Done"

        # masker for SpeakerBeam expects (tf_rep, enroll_emb)
        est_masks = self.base.forward_masker(tf_rep, z)
        masked_tf_rep = self.base.apply_masks(tf_rep, est_masks)
        decoded = self.base.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)

    def serialize(self):
        import torch
        # Serialize the base model and add wrapper info
        data = self.base.serialize()
        data["wrapper_config"] = {
            "spk_dim": self.spk_dim,
            "text_model_name": self.text_model_name,
            "use_lora": self.use_lora,
            "allow_text_only": self.allow_text_only,
            "fusion_type": self.fusion_type
        }
        data["state_dict"] = self.state_dict()
        return data

    def get_config(self):
        return {
            "spk_dim": self.spk_dim,
            "text_model_name": self.text_model_name,
            "use_lora": self.use_lora,
            "allow_text_only": self.allow_text_only,
            "fusion_type": self.fusion_type
        }