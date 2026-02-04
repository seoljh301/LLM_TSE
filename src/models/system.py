# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from asteroid.engine import System

class SystemInformed(System):
    def common_step(self, batch, batch_nb, train=True):
        # NMDataset returns: mix, tgt, enroll, text, gt_sentence
        if len(batch) == 5:
            inputs, targets, enrolls, texts, _ = batch
            # enrolls will be None because return_enroll=False in NMDataset
            est_targets = self(inputs, enrolls, texts=texts)
        else:
            # Fallback for standard 3-element batches
            inputs, targets, enrolls = batch
            est_targets = self(inputs, enrolls)
            
        loss = self.loss_func(est_targets, targets)
        return loss
