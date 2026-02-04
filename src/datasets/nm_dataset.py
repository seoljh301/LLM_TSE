import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
import random
import os
import numpy as np

class NMDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, segment=3.0, return_enroll=True):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment = segment
        self.return_enroll = return_enroll
        
        # Check if train or test
        if "train" in data_dir:
            self.split = "train"
        elif "test" in data_dir:
            self.split = "test"
        else:
            self.split = "train" # Default

        # Detect version from data_dir (e.g., nm_v16 -> v16)
        import re
        match = re.search(r"nm_(v\d+)", data_dir)
        version = match.group(1) if match else "v15"
            
        csv_2sp = os.path.join(data_dir, f"nm_{version}_{self.split}_2sp.csv")
        csv_path = os.path.join(data_dir, f"nm_path_{version}_{self.split}.csv")
        
        self.df_mix = pd.read_csv(csv_2sp)
        self.df_path = pd.read_csv(csv_path)
        
        # Merge on file_id
        # df_mix has [file_id, prompt, mixed_audio_path, t_speaker_id, ...]
        # df_path has [file_id, t_speaker_id, target_audio_path, ...]
        self.df = pd.merge(self.df_mix, self.df_path, on="file_id", suffixes=('', '_path'))
        
        # Build speaker index for enrollment (using t_speaker_id from the mix csv)
        # target_audio_path comes from path csv
        self.speakers = self.df.groupby("t_speaker_id")["target_audio_path"].apply(list).to_dict()
        
    def __len__(self):
        return len(self.df)
        
    def _crop(self, audio):
        if self.segment is None:
            return audio
        tgt_len = int(self.segment * self.sample_rate)
        src_len = audio.shape[-1]
        if src_len <= tgt_len:
            # Pad
            if len(audio.shape) == 1:
                return np.pad(audio, (0, tgt_len - src_len))
            else:
                return np.pad(audio, ((0,0), (0, tgt_len - src_len)))
        else:
            # Random crop
            start = random.randint(0, src_len - tgt_len)
            if len(audio.shape) == 1:
                return audio[start:start+tgt_len]
            else:
                return audio[:, start:start+tgt_len]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Mixture
        mix_path = row["mixed_audio_path"]
        mix, _ = sf.read(mix_path, dtype="float32")
        if len(mix.shape) == 2:
            mix = mix.T # (C, T)
            
        # Load Target
        tgt_path = row["target_audio_path"]
        tgt, _ = sf.read(tgt_path, dtype="float32")
        if len(tgt.shape) == 2:
            tgt = tgt.T
            
        # Crop logic (Synchronous)
        if self.segment is not None:
            tgt_len = int(self.segment * self.sample_rate)
            src_len = mix.shape[-1]
            if src_len > tgt_len:
                start = random.randint(0, src_len - tgt_len)
                stop = start + tgt_len
                # Handle 1D vs 2D cropping
                if len(mix.shape) == 1:
                    mix = mix[start:stop]
                    tgt = tgt[start:stop]
                else:
                    mix = mix[:, start:stop]
                    tgt = tgt[:, start:stop]
            else:
                 # Pad
                pad_len = tgt_len - src_len
                if len(mix.shape) == 1:
                    mix = np.pad(mix, (0, pad_len))
                    tgt = np.pad(tgt, (0, pad_len))
                else:
                    mix = np.pad(mix, ((0,0), (0, pad_len)))
                    tgt = np.pad(tgt, ((0,0), (0, pad_len)))

        # Enrollment
        enroll = torch.tensor([]) # Default to empty tensor instead of None
        if self.return_enroll:
            spk_id = row["t_speaker_id"]
            opts = self.speakers.get(spk_id, [])
            cands = [p for p in opts if p != tgt_path]
            if not cands: cands = [tgt_path]
            
            # Robustness: if cands empty (shouldn't be if tgt_path in opts), use tgt_path
            if not cands:
                 enroll_path = tgt_path
            else:
                 enroll_path = random.choice(cands)
            
            enroll_wav, _ = sf.read(enroll_path, dtype="float32")
            if len(enroll_wav.shape) == 2:
                enroll_wav = enroll_wav.T
            # Crop enrollment independently
            enroll = self._crop(enroll_wav)
            enroll = torch.from_numpy(enroll)
            
        # Text
        text = row["prompt"]
        
        # Ground Truth Sentence (for WER)
        gt_sentence = row["target_sentence"] if "target_sentence" in row else ""
        
        return torch.from_numpy(mix), torch.from_numpy(tgt), enroll, text, gt_sentence

    def get_infos(self):
        return {}
