# SpeakerBeam for neural target speech extraction

This repository contains an implementation of SpeakerBeam method for target speech extraction, made public during Interspeech 2021 tutorial.

The code is based on the [Asteroid toolkit](https://github.com/asteroid-team/asteroid) for audio speech separation.

## Requirements

To install requirements:
```
pip install -r requirements.txt
```
The code was tested with Python 3.8.6.

## Running the experiments
While this repository provides a recipe for the Libri2mix dataset, current experiments are conducted using a custom dataset named **PORTE**. 

**Note:** The PORTE dataset is currently private and will be publicly released in the future.

### Environment Setup
Before running any scripts, ensure the repository root is in your `PYTHONPATH`. You can configure `path.sh`:
```bash
# In path.sh
PATH_TO_REPOSITORY="$(pwd)" # Update to the absolute path of this repo
export PYTHONPATH=${PATH_TO_REPOSITORY}/src:$PYTHONPATH
```
Then source it:
```bash
source path.sh
```
Alternatively, set it directly in your shell:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Preparing data
For the Libri2mix recipe, use:
```bash
cd egs/libri2mix
local/prepare_data.sh <path-to-libri2mix-data>
```
The `<path-to-libri2mix-data>` should contain `wav8k/min` subdirectories.

For the **PORTE** dataset (consisting of versions like `nm_v16/train` and `nm_v15/test`), the directory structure is organized as follows:

```text
PORTE_dataset/
├── add/                # Enrollment/Auxiliary speech signals
├── mixed/              # Mixture speech signals (input)
├── trg/                # Target speech signals (ground truth)
├── nm_path_v1X_*.csv   # CSV file containing absolute paths to audio files
└── nm_v1X_*_2sp.csv    # CSV file containing metadata for 2-speaker mixtures
```

Ensure these metadata CSV files are present in your `--test_dir` or `--train_dir`.

### Training SpeakerBeam
To train the SpeakerBeam model:
```bash
# Ensure you are in egs/libri2mix and PYTHONPATH is set
python train.py --exp_dir exp/speakerbeam
```
Default parameters are in `local/conf.yml`. You can override them via CLI:
```bash
python train.py --exp_dir exp/speakerbeam_adaptlay15 --i_adapt_layer 15
```

### Decoding and Evaluation
To evaluate a trained model or a specific checkpoint on the test set:
```bash
python eval.py \
  --test_dir /path/to/PORTE/test_data \
  --task sep_noisy \
  --model_path exp/speakerbeam/checkpoints/epoch=25-step=146250.ckpt \
  --from_checkpoint 1 \
  --out_dir exp/speakerbeam/out_test \
  --exp_dir exp/speakerbeam \
  --use_gpu 1
```

In the output directory, `final_metrics.json` contains the averaged results, and extracted audio files are saved in the `out/` subdirectory.
