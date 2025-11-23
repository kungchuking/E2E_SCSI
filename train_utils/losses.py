# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

# -- Added by Chu King on 23rd November 2025 to check for NaNs
import math

import matplotlib.pyplot as plt
import numpy as np

def flow_to_rgb(flow):
    # flow: [2, H, W]
    u = flow[0]
    v = flow[1]
    rad = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(rad / np.percentile(rad, 99), 0, 1)

    rgb = plt.cm.hsv(hsv)
    return rgb[..., :3]

def visualize_flow_debug(flow_pred, flow_gt, epe, step=0, save_path="debug"):
    flow_pred_np = flow_pred.detach().cpu().numpy()
    flow_gt_np   = flow_gt.detach().cpu().numpy()
    epe_np       = epe.detach().cpu().numpy()

    flow_pred0 = flow_pred_np[0, 0, :, :]
    flow_gt0   = flow_gt_np[0, 0, :, :]
    epe0       = epe_np[0, 0, :, :]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(flow_to_rgb(flow_pred0))
    axs[0].set_title("Predicted Flow")
    axs[0].axis("off")

    axs[1].imshow(flow_to_rgb(flow_gt0))
    axs[1].set_title("Ground Truth Flow")
    axs[1].axis("off")

    axs[2].imshow(epe0, cmap="inferno")
    axs[2].set_title("EPE heatmap")
    axs[2].axis("off")

    fig.suptitle(f"STEP = {step}")

    plt.tight_layout()
    plt.savefig(f"{save_path}/flow_debug_{step}.png")
    plt.close()

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """Loss function defined over sequence of flow predictions"""
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt().unsqueeze(1)

    if len(valid.shape) != len(flow_gt.shape):
        valid = valid.unsqueeze(1)

    valid = (valid >= 0.5) & (mag < max_flow)

    if valid.shape != flow_gt.shape:
        valid = torch.cat([valid, valid], dim=1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )

        if n_predictions == 1:
            i_weight = 1
        else:
            # We adjust the loss_gamma so it is consistent for any number of iterations
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

        flow_pred = flow_preds[i].clone()
        if valid.shape[1] == 1 and flow_preds[i].shape[1] == 2:
            flow_pred = flow_pred[:, :1]

        i_loss = (flow_pred - flow_gt).abs()

        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            flow_gt.shape,
            flow_pred.shape,
        ]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    valid = valid[:, 0]
    epe = epe.view(-1)
    epe = epe[valid.reshape(epe.shape)]

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    for k, v in metrics.items():
        if math.isnan(v):
            print ("[ERROR] Nan detected for k: ", k)
            if torch.isnan(flow_preds[-1]).any(): print("[WARNING] NaN in flow_preds")
            if torch.isinf(flow_preds[-1]).any(): print("[WARNING] Inf in flow_preds")
            if torch.isnan(flow_gt).any(): print("[WARNING] NaN in flow_gt")
            if torch.isinf(flow_gt).any(): print("[WARNING] Inf in flow_gt")

            raw_diff = flow_preds[-1] - flow_gt
            if torch.isnan(raw_diff).any(): print("[WARNING] NaN in flow_diff")
            
            sq = (raw_diff ** 2)
            if torch.isnan(sq).any(): print("[WARNING] NaN in square")
            
            sum_sq = torch.sum(sq, dim=1)
            if torch.isnan(sum_sq).any(): print("[WARNING] NaN in sum")
            
            epe = sum_sq.sqrt()
            if torch.isnan(epe).any(): print("[WARNING] NaN in sqrt")
            if torch.isinf(epe).any(): print("[WARNING] Inf in sqrt")

            num_valid = valid.sum().item()
            print("[INFO] Valid pixels:", num_valid)
            if num_valid == 0:
                print("[WARNING]: No valid pixels  metrics will be NaN.")

            if (epe > 1e6).any():
                print("[INFP] Large EPE values detected:", epe.max().item())

            print ("[INFO] Flow pred sample:", flow_preds[-1].view(-1)[:10])
            print ("[INFO] Flow gt sample:", flow_gt.view(-1)[:10])
            print ("[INFO] EPE sample:", epe.view(-1)[:10])
            print ("[INFO] Valid sample:", valid.view(-1)[:10])

            visualize_flow_debug(flow_preds[-1], flow_gt, v, step=0, save_path="debug")
            raise SystemExit("Nan detected.")

    return flow_loss, metrics
