import os
import argparse
import json
import yaml
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import DataLoader
from asteroid.losses import singlesrc_neg_sisdr
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality
from rich.console import Console
from rich.progress import track
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys
from pathlib import Path

# Setup paths to ensure src is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

# Import dataset and model classes
from src.datasets.nm_dataset import NMDataset
from models.td_speakerbeam import TimeDomainSpeakerBeam
from src.llmtse_wrapper import LLMTSEWrapper

def calculate_wer(reference, hypothesis):
    import jiwer
    return jiwer.wer(reference, hypothesis)

def main(conf):
    console = Console()
    console.rule("[bold green]LLM-TSE Comprehensive Evaluation[/]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    exp_dir = conf["exp_dir"]
    checkpoint_path = conf.get("ckpt_path")
    
    if not checkpoint_path:
        checkpoint_path = os.path.join(exp_dir, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        console.print(f"[red]Error: Checkpoint not found at {checkpoint_path}[/]")
        return

    console.print(f"[blue]Loading TSE model from {checkpoint_path}...[/]")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load hyperparams from conf.yml saved during training
    train_conf_path = os.path.join(exp_dir, "conf.yml")
    if os.path.exists(train_conf_path):
        with open(train_conf_path) as f:
            train_conf = yaml.safe_load(f)
    else:
        # Fallback to default PORTE-like config if missing
        console.print("[yellow]Warning: conf.yml not found. Using PORTE-based defaults.[/]")
        train_conf = {
            "data": {"sample_rate": 16000, "valid_dir": "/nfs/data/speech-data/nm/nm_v15/test"},
            "filterbank": {"n_filters": 512, "kernel_size": 16, "stride": 8},
            "masknet": {"n_src": 2, "n_blocks": 8, "n_repeats": 3, "mask_act": "relu", "bn_chan": 128, "hid_chan": 512, "skip_chan": 128},
            "enroll": {"i_adapt_layer": 7, "adapt_layer_type": "mul", "adapt_enroll_dim": 128}
        }

    # Instantiate Model
    base_model = TimeDomainSpeakerBeam(
        **train_conf["filterbank"], 
        **train_conf["masknet"], 
        sample_rate=train_conf["data"]["sample_rate"],
        **train_conf["enroll"]
    )
    
    # Wrapper config (Try to get from checkpoint, else infer)
    wrapper_conf = checkpoint.get("wrapper_config", {})
    spk_dim = train_conf["enroll"]["adapt_enroll_dim"]
    if train_conf["masknet"].get("skip_chan", 0) > 0:
        spk_dim *= 2
        
    # Use wrapper config or fallback to defaults (Note: LoRA should be checked)
    use_lora = wrapper_conf.get("use_lora", True) # Default to True as we changed train.py
    
    model = LLMTSEWrapper(
        base_model, 
        spk_dim=wrapper_conf.get("spk_dim", spk_dim),
        text_model_name=wrapper_conf.get("text_model_name", "meta-llama/Llama-2-7b-hf"),
        use_lora=use_lora
    )
    
    state_dict = checkpoint["state_dict"]
    # Handle Lightning state dict (strip "model." prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
        else:
            new_state_dict[k] = v
            
    # Load weights (strict=False because Llama might have buffer mismatches or LoRA keys)
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        console.print(f"[yellow]Warning loading state_dict: {e}[/]")
        
    model.eval()
    model.to(device)
    
    # 2. Init Metrics
    console.print("[blue]Initializing Metrics (STOI, PESQ, Whisper-v3)...[/]")
    sample_rate = train_conf["data"]["sample_rate"] # 16000
    
    # Metrics
    stoi_metric = ShortTimeObjectiveIntelligibility(sample_rate, extended=False).to(device)
    pesq_metric = PerceptualEvaluationSpeechQuality(sample_rate, 'wb').to(device) # 'wb' for 16k
    
    # ASR (Whisper-v3)
    try:
        asr_model_id = "openai/whisper-large-v3"
        # Use torch.float16 if GPU, else float32
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        asr_model.to(device)
        processor = AutoProcessor.from_pretrained(asr_model_id)
        
        console.print("[green]Whisper-v3 loaded for WER evaluation (Direct Inference).[/]")
        
        # Install jiwer if missing
        import jiwer
    except ImportError:
        console.print("[yellow]jiwer not found. Installing...[/]")
        os.system("pip install jiwer")
        import jiwer
    except Exception as e:
        console.print(f"[red]Failed to load Whisper: {e}. Skipping WER.[/]")
        asr_model = None

    # 3. Load Test Data
    test_dir = conf.get("test_dir")
    if not test_dir:
        test_dir = train_conf["data"].get("valid_dir", "/nfs/data/speech-data/nm/nm_v15/test")

    if not os.path.exists(test_dir):
        console.print(f"[red]Error: Test directory not found at {test_dir}[/]")
        return

    console.print(f"[blue]Loading test data from {test_dir}...[/]")
    test_set = NMDataset(
        data_dir=test_dir,
        sample_rate=sample_rate,
        segment=None, # Evaluate full
        return_enroll=True
    )
    
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=1, 
        num_workers=4
    )

    # 4. Loop
    results = []
    save_dir = os.path.join(exp_dir, "examples")
    os.makedirs(save_dir, exist_ok=True)
    
    # Pick 10 random indices to save
    import random
    total_samples = len(test_loader)
    save_indices = set(random.sample(range(total_samples), k=min(10, total_samples)))
    console.print(f"[blue]Saving audio for indices: {save_indices}[/]")
    
    console.print("[bold yellow]Starting Evaluation Loop...[/]")
    
    with torch.no_grad():
        for i, batch in track(enumerate(test_loader), total=len(test_loader), description="Eval"):
            mix, tgt, enroll, text, gt_sentence = batch
            mix, tgt, enroll = mix.to(device), tgt.to(device), enroll.to(device)
            
            # Forward
            est = model(mix, enroll, texts=text)
            
            # Fix shapes: (B, T)
            if est.ndim == 3: est = est.squeeze(1)
            if tgt.ndim == 3: tgt = tgt.squeeze(1)
            if mix.ndim == 3: mix = mix.squeeze(1)
            
            # 1. SI-SDR & SI-SDRi (with Silent Check)
            if torch.sum(tgt**2) < 1e-7:
                sisdr = float('nan')
                sisdr_i = float('nan')
            else:
                # Estimated SI-SDR
                loss_est = singlesrc_neg_sisdr(est, tgt)
                sisdr = -loss_est.item()
                
                # Mixture SI-SDR (Baseline)
                loss_mix = singlesrc_neg_sisdr(mix, tgt)
                sisdr_mix = -loss_mix.item()
                sisdr_i = sisdr - sisdr_mix
            
            # 2. STOI
            try:
                val_stoi = stoi_metric(est, tgt).item()
            except:
                val_stoi = float('nan')
            
            # 3. PESQ (Requires 16k or 8k)
            try:
                val_pesq = pesq_metric(est, tgt).item()
            except Exception:
                val_pesq = float('nan') 
                
            # 4. WER (Compare Est with Ground Truth)
            wer = float('nan')
            est_text = ""
            ref_text = gt_sentence[0] if gt_sentence else ""
            
            if asr_model and ref_text:
                est_np = est[0].cpu().numpy().astype(np.float32)
                
                # Preprocess & Generate
                input_features_est = processor(est_np, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device).to(torch_dtype)
                pred_ids_est = asr_model.generate(input_features_est, language="en")
                est_text = processor.batch_decode(pred_ids_est, skip_special_tokens=True)[0]
                
                # Calculate WER against GT
                try:
                    wer = jiwer.wer(ref_text, est_text)
                except:
                    wer = 1.0 

            results.append({
                "id": i,
                "si_sdr": sisdr,
                "si_sdri": sisdr_i,
                "stoi": val_stoi,
                "pesq": val_pesq,
                "wer": wer,
                "prompt": text[0],
                "ref_text": ref_text,
                "hyp_text": est_text
            })
            
            # Save examples (Random 10)
            if i in save_indices:
                est_audio = est[0].cpu().numpy()
                tgt_audio = tgt[0].cpu().numpy()
                
                # Normalize to prevent clipping
                max_est = np.abs(est_audio).max()
                if max_est > 0.99:
                    est_audio = est_audio / max_est * 0.99
                    
                max_tgt = np.abs(tgt_audio).max()
                if max_tgt > 0.99:
                    tgt_audio = tgt_audio / max_tgt * 0.99
                
                sf.write(os.path.join(save_dir, f"ex_{i}_est.wav"), est_audio, sample_rate)
                sf.write(os.path.join(save_dir, f"ex_{i}_tgt.wav"), tgt_audio, sample_rate)
                with open(os.path.join(save_dir, f"ex_{i}_meta.txt"), "w") as f:
                    f.write(f"Prompt: {text[0]}\nRef: {ref_text}\nHyp: {est_text}\nWER: {wer}\nSDR: {sisdr}\nSDRi: {sisdr_i}")

    # 5. Summary
    df = pd.DataFrame(results)
    console.print("\n[bold]Final Results:[/]")
    console.print(f"Mean SI-SDR:  {df['si_sdr'].mean():.2f} dB")
    console.print(f"Mean SI-SDRi: {df['si_sdri'].mean():.2f} dB")
    console.print(f"Mean STOI:    {df['stoi'].mean():.3f}")
    console.print(f"Mean PESQ:    {df['pesq'].mean():.3f}")
    console.print(f"Mean WER:     {df['wer'].mean():.3f}")
    
    df.to_csv(os.path.join(exp_dir, "test_metrics.csv"), index=False)
    console.print(f"[blue]Detailed metrics saved to {os.path.join(exp_dir, 'test_metrics.csv')}[/]")

if __name__ == "__main__":
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment root")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint file (.ckpt or .pth)")
    args = parser.parse_args()
    
    arg_dic = vars(args)
    if arg_dic["exp_dir"] is None:
        # Find the latest directory in 'exp'
        exp_dirs = [d for d in glob.glob("exp/*") if os.path.isdir(d)]
        if not exp_dirs:
            print("Error: No directories found in 'exp/'. Please provide --exp_dir.")
            sys.exit(1)
        # Sort by modification time (latest first)
        latest_dir = max(exp_dirs, key=os.path.getmtime)
        arg_dic["exp_dir"] = latest_dir
        print(f"No --exp_dir provided. Automatically picking the latest: {latest_dir}")
        
    main(arg_dic)