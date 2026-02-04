import os
import argparse
import json
import yaml
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

# Setup paths to ensure src is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from models.td_speakerbeam import TimeDomainSpeakerBeam
from src.datasets.nm_dataset import NMDataset
from src.llmtse_wrapper import LLMTSEWrapper
from asteroid.engine.optimizers import make_optimizer
from models.system import SystemInformed
from asteroid.losses import singlesrc_neg_sisdr
from datetime import datetime

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs.
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default=None, help="Full path to save best validation model")

def neg_sisdr_loss_wrapper(est_targets, targets):
    # est_targets is usually (B, 1, T) or (B, T)
    if est_targets.ndim == 3:
        est_targets = est_targets.squeeze(1)
    if targets.ndim == 3:
        targets = targets.squeeze(1)
    return singlesrc_neg_sisdr(est_targets, targets).mean()

def main(conf):
    train_set = NMDataset(
        data_dir=conf["data"]["train_dir"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        return_enroll=False
    )

    val_set = NMDataset(
        data_dir=conf["data"]["valid_dir"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        return_enroll=False
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    # Initialize Base Model
    base_model = TimeDomainSpeakerBeam(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"],
        **conf["enroll"]
    )
    
    # Wrap with LLMTSEWrapper
    # Note: These parameters can be added to conf.yml if needed
    spk_dim = conf["enroll"]["adapt_enroll_dim"]
    if conf["masknet"].get("skip_chan", 0) > 0:
        spk_dim *= 2

    model = LLMTSEWrapper(
        base_model,
        spk_dim=spk_dim,
        text_model_name=conf.get("text_model", "meta-llama/Llama-2-7b-chat-hf"),
        use_lora=conf.get("use_lora", True),
        load_in_4bit=conf.get("load_in_4bit", False),
        fusion_type=conf.get("fusion_type", "concat")
    )

    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=conf['training']['reduce_patience'])
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    if exp_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join("exp", timestamp)
        conf["main_args"]["exp_dir"] = exp_dir
        print(f"No exp_dir provided. Automatically creating: {exp_dir}")

    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = neg_sisdr_loss_wrapper
    system = SystemInformed(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=-1, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=conf['training']['stop_patience'], verbose=True))
    callbacks.append(LearningRateMonitor())

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    
    # For newer PyTorch Lightning versions
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 # Change as needed

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # Save best model
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    
    # Save using the wrapper's serialize method
    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
