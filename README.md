# LLM-TSE: LLM-based Target Speech Extraction

This repository contains an implementation of **LLM-TSE**, which extends the SpeakerBeam method to incorporate LLM-based text conditioning for target speech extraction.

The code is based on the [Asteroid toolkit](https://github.com/asteroid-team/asteroid) and utilizes a pre-trained Llama-2 model via LoRA for advanced text-to-audio conditioning.

## Requirements

To install requirements:
```bash
pip install -r requirements.txt
```
The code was tested with Python 3.10 and requires PyTorch with CUDA support for LLM inference.

## Project Structure
All core scripts are located in the root directory for ease of use:
- `train.py`: Main training script for LLM-TSE.
- `test.py`: Comprehensive evaluation script including SI-SDR, STOI, PESQ, and WER (via Whisper-v3).
- `analyze_results.py`: Post-evaluation analysis grouped by overlap ratio.
- `src/`: Core model architecture (LLMTSEWrapper, FiLMFusion, etc.) and datasets.
- `local/conf.yml`: Default configuration for experiments.

## Running the experiments
Currently, experiments are conducted using the **PORTE** custom dataset.

**Note:** The PORTE dataset is currently private and will be publicly released in the future.

### Dataset Structure (PORTE)
For PORTE dataset versions (e.g., `nm_v16/train` and `nm_v15/test`), the directory structure is:

```text
PORTE_dataset/
├── add/                # Enrollment/Auxiliary speech signals
├── mixed/              # Mixture speech signals (input)
├── trg/                # Target speech signals (ground truth)
├── nm_path_v1X_*.csv   # CSV file with absolute paths
└── nm_v1X_*_2sp.csv    # Metadata for 2-speaker mixtures (contains overlap_ratio)
```

### Environment Setup
Add the repository root to your `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Training LLM-TSE
To train the model using the default configuration (`local/conf.yml`):
```bash
python train.py --exp_dir exp/llmtse_v1
```
You can override parameters like the fusion type or LoRA usage:
```bash
python train.py --exp_dir exp/llmtse_film --fusion_type film --use_lora 1
```

### Evaluation
To evaluate a specific checkpoint on the test set:
```bash
python test.py \
  --exp_dir exp/llmtse_v1 \
  --ckpt_path exp/llmtse_v1/checkpoints/best_model.pth
```
This script will calculate SI-SDR, STOI, PESQ, and WER. Audio examples will be saved in `exp/llmtse_v1/examples`.

### Analysis
To generate a performance table grouped by overlap ratio:
```bash
python analyze_results.py --exp_dir exp/llmtse_v1
```
Results will be saved in `analysis_by_overlap.csv`.
