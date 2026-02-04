# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from asteroid.data import LibriMix
import random
import torch
import soundfile as sf
import os

def read_enrollment_csv(csv_path):
    data = defaultdict(dict)
    with open(csv_path, 'r') as f:
        f.readline() # csv header

        for line in f:
            mix_id, utt_id, *aux = line.strip().split(',')
            aux_it = iter(aux)
            aux = [(auxpath,int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
            data[mix_id][utt_id] = aux
    return data

class LibriMixInformed(Dataset):
    def __init__(
        self, csv_dir, task="sep_clean", sample_rate=16000, n_src=2, 
        segment=3, segment_aux=3, 
        ):
        
        # Check for standard LibriMix CSV
        std_csv = Path(csv_dir) / f"mixture_{task}.csv"
        
        if std_csv.exists():
            self.base_dataset = LibriMix(csv_dir, task, sample_rate, n_src, segment)
            self.data_aux = read_enrollment_csv(Path(csv_dir) / 'mixture2enrollment.csv')
            self.is_custom = False
        else:
            print(f"Standard LibriMix CSV not found in {csv_dir}. Checking for custom NM dataset...")
            # Custom NM Dataset Logic
            path_csv = Path(csv_dir) / "nm_path_v15_test.csv"
            meta_csv = Path(csv_dir) / "nm_v15_test_2sp.csv"
            
            if path_csv.exists() and meta_csv.exists():
                df1 = pd.read_csv(meta_csv)
                df2 = pd.read_csv(path_csv)
                # Merge on file_id
                self.df = pd.merge(df1, df2, on='file_id', suffixes=('', '_y'))
                self.df['mixture_ID'] = self.df['file_id']
                
                # Mock base dataset
                class MockBase:
                    pass
                self.base_dataset = MockBase()
                self.base_dataset.df = self.df
                self.base_dataset.seg_len = int(segment * sample_rate) if segment is not None else None
                
                self.is_custom = True
                
                # Create dummy enrollment map (Self-enrollment using target audio)
                self.data_aux = defaultdict(dict)
                for _, row in self.df.iterrows():
                    # Assuming target_audio_path is available and valid
                    # We use a large dummy length to ensure it's picked if check is loose, 
                    # or we should check file length if possible. For now, hardcode large.
                    self.data_aux[row['file_id']][row['file_id']] = [(row['target_audio_path'], 16000*30)]
            else:
                 # Fallback to attempting standard load which might fail or raise error
                 self.base_dataset = LibriMix(csv_dir, task, sample_rate, n_src, segment)
                 self.data_aux = read_enrollment_csv(Path(csv_dir) / 'mixture2enrollment.csv')
                 self.is_custom = False

        if segment_aux is not None:
            max_len = np.sum([len(self.data_aux[m][u]) for m in self.data_aux 
                                                     for u in self.data_aux[m]])
            self.seg_len_aux = int(segment_aux * sample_rate)
            self.data_aux = {m: {u:  
                [(path,length) for path, length in self.data_aux[m][u]
                    if length >= self.seg_len_aux
                    ]
                for u in self.data_aux[m]} for m in self.data_aux}
            new_len = np.sum([len(self.data_aux[m][u]) for m in self.data_aux 
                                                     for u in self.data_aux[m]])
            print(
                f"Drop {max_len - new_len} utterances from {max_len} "
                f"(shorter than {segment_aux} seconds)"
            )
        else:
            self.seg_len_aux = None

        self.seg_len = self.base_dataset.seg_len

        # to choose pair of mixture and target speaker by index
        self.data_aux_list = [(m,u) for m in self.data_aux 
                                    for u in self.data_aux[m]]

    def __len__(self):
        return len(self.data_aux_list)

    def _get_segment_start_stop(self, seg_len, length):
        if seg_len is not None:
            if length < seg_len:
                start = 0
                stop = length # Or pad? Original code implies length > seg_len usually
            else:
                start = random.randint(0, length - seg_len)
                stop = start + seg_len
        else:
            start = 0
            stop = None
        return start, stop

    def __getitem__(self, idx):
        mix_id, utt_id = self.data_aux_list[idx]
        row = self.base_dataset.df[self.base_dataset.df['mixture_ID'] == mix_id].squeeze()

        # Handle Custom Paths vs Standard
        if self.is_custom:
            mixture_path = row['mixed_audio_path']
            self.mixture_path = mixture_path
            self.target_speaker_idx = 0 # Dummy for custom
            source_path = row['target_audio_path']
            length = 16000 * 30 # Dummy large length
        else:
            mixture_path = row['mixture_path']
            self.mixture_path = mixture_path
            tgt_spk_idx = mix_id.split('_').index(utt_id)
            self.target_speaker_idx = tgt_spk_idx
            source_path = row[f'source_{tgt_spk_idx+1}_path']
            length = row['length']

        # read mixture
        start, stop = self._get_segment_start_stop(self.seg_len, length)
        mixture,_ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        mixture = torch.from_numpy(mixture)

        # read source
        source,_ = sf.read(source_path, dtype="float32", start=start, stop=stop)
        source = torch.from_numpy(source)[None]

        # read enrollment
        enroll_path, enroll_length = random.choice(self.data_aux[mix_id][utt_id])
        start_e, stop_e = self._get_segment_start_stop(self.seg_len_aux, enroll_length)
        enroll,_ = sf.read(enroll_path, dtype="float32", start=start_e, stop=stop_e)
        enroll = torch.from_numpy(enroll)

        return mixture, source, enroll

    def get_infos(self):
        return self.base_dataset.get_infos()

