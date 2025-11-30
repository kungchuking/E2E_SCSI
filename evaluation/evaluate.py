# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append("../")

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim

from munch import DefaultMunch
import json
from pytorch_lightning.lite import LightningLite
from torch.cuda.amp import GradScaler

from train_utils.utils import (
    run_test_eval,
    save_ims_to_tb,
    count_parameters,
)
from train_utils.logger import Logger
from models.core.dynamic_stereo import DynamicStereo
from models.core.sci_codec import sci_encoder
from evaluation.core.evaluator import Evaluator
from train_utils.losses import sequence_loss
import datasets.dynamic_stereo_datasets as datasets

class wrapper(nn.Module):
    def __init__(
            self, 
            sigma_range=[0, 1e-9],
            num_frames=8,
            in_channels=1,
            n_taps=2,
            resolution=[480, 640],
            mixed_precision=True,
            attention_type="self_stereo_temporal_update_time_update_space",
            update_block_3d=True,
            different_update_blocks=True,
            train_iters=16):

        super(wrapper, self).__init__()

        self.train_iters = train_iters

        self.sci_enc_L = sci_encoder(sigma_range=sigma_range,
                                     n_frame=num_frames,
                                     in_channels=in_channels,
                                     n_taps=n_taps,
                                     resolution=resolution)
        self.sci_enc_R = sci_encoder(sigma_range=sigma_range,
                                     n_frame=num_frames,
                                     in_channels=in_channels,
                                     n_taps=n_taps,
                                     resolution=resolution)

        self.stereo = DynamicStereo(max_disp=256,
                                    mixed_precision=mixed_precision,
                                    num_frames=num_frames,
                                    attention_type=attention_type,
                                    use_3d_update_block=update_block_3d,
                                    different_update_blocks=different_update_blocks)

    def forward(self, batch):
        # ---- ---- FORWARD PASS ---- ----
        # -- Modified by Chu King on 20th November 2025
        
        # -- print ("[INFO] batch[\"img\"].device: ", batch["img"].device)

        # 0) Convert to Gray
        def rgb_to_gray(x):
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=x.dtype, device=x.device)
            gray = (x * weights[None, None, :, None, None]).sum(dim=2)
            return gray # -- shape: [B, T, H, W]
        
        video_L = rgb_to_gray(batch["img"][:, :, 0]) # ~ (b, t, h, w)
        video_R = rgb_to_gray(batch["img"][:, :, 1]) # ~ (b, t, h, w)

        # -- print ("[INFO] video_L.device: ", video_L.device)
        
        # 1) Extract and normalize input videos.
        # -- min_max_norm = lambda x : 2. * (x / 255.) - 1.
        min_max_norm = lambda x: x / 255.
        video_L = min_max_norm(video_L) # ~ (b, t, h, w)
        video_R = min_max_norm(video_R) # ~ (b, t, h, w)
        # -- print ("[INFO] video_L.device: ", video_L.device)
        
        # 2) If the tensor is non-contiguous and we try .view() later, PyTorch will raise an error:
        video_L = video_L.contiguous()
        video_R = video_R.contiguous()

        # -- print ("[INFO] video_L.device: ", video_L.device)
        
        # 3) Coded exposure modeling.
        snapshot_L = self.sci_enc_L(video_L) # ~ (b, c, h, w) -- c=2 for 2 taps
        snapshot_R = self.sci_enc_R(video_R) # ~ (b, c, h, w) -- c=2 for 2 taps

        # -- print ("[INFO] self.sci_enc_L.device: ", next(self.sci_enc_R.parameters()).device)
        # -- print ("[INFO] snapshot_L.device: ", snapshot_L.device)
        
        # 4) Dynamic Stereo
        output = {}
        
        disparities = self.stereo(
            snapshot_L,
            snapshot_R,
            iters=self.train_iters,
            test_mode=False
        )
        
        n_views = len(batch["disp"][0]) # -- sample_len
        for i in range(n_views):
            seq_loss, metrics = sequence_loss(
                disparities[:, i], batch["disp"][:, i, 0], batch["valid_disp"][:, i, 0]
            )

            output[f"disp_{i}"] = {"loss": seq_loss / n_views, "metrics": metrics}
        output["disparity"] = {
            "predictions": torch.cat(
                [disparities[-1, i, 0] for i in range(n_views)], dim=1
            ).detach(),
        }
        return output

if __name__ == "__main__":
    eval_dataloader_dr = datasets.DynamicReplicaDataset(
        split="valid", sample_len=8, only_first_n_samples=1, VERBOSE=False, root="../dynamic_replica_data", t_step_validation=4
    )
    
    eval_dataloader_sintel_clean = datasets.SequenceSintelStereo(dstype="clean")
    eval_dataloader_sintel_final = datasets.SequenceSintelStereo(dstype="final")
    
    eval_dataloaders = [
        ("sintel_clean", eval_dataloader_sintel_clean),
        ("sintel_final", eval_dataloader_sintel_final),
        ("dynamic_replica", eval_dataloader_dr),
    ]
    
    evaluator = Evaluator()
    
    eval_vis_cfg = {
        "visualize_interval": 1,  # Use 0 for no visualization
        "exp_dir": "./"
    }
    eval_vis_cfg = DefaultMunch.fromDict(eval_vis_cfg, object())
    evaluator.setup_visualization(eval_vis_cfg)
    
    # ----------------------------------------- Model Instantiation -----------------------------------------------
    model = wrapper(sigma_range=[0, 1e-9],
                    num_frames=8,
                    in_channels=1,
                    n_taps=2,
                    resolution=[480, 640],
                    mixed_precision=True,
                    attention_type="self_stereo_temporal_update_time_update_space",
                    update_block_3d=True,
                    different_update_blocks=True,
                    train_iters=8)

    ckpt_path = "../dynamicstereo_sf_dr/model_dynamic-stereo_050895.pth"
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()

    run_test_eval(
        ckpt_path="./",
        eval_type="valid",
        evaluator=evaluator,
        sci_enc_L=model.sci_enc_L,
        sci_enc_R=model.sci_enc_R,
        model=model.stereo,
        dataloaders=eval_dataloaders,
        writer=None,
        step=None,
        resolution=[480, 640]
    )
    

