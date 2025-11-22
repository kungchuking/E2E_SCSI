# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
        
        video_L = rgb_to_gray(batch["img"][:, :, 0]).cuda() # ~ (b, t, h, w)
        video_R = rgb_to_gray(batch["img"][:, :, 1]).cuda() # ~ (b, t, h, w)

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

        n_views = len(batch["disp"][0])
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

def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


# -- Modified by Chu King on 20th November 2025
# -- Take snapshots instead of videos as input.
# -- def forward_batch(batch, model, args):
def forward_batch(snapshot_L, snapshot_R, model, args):
    output = {}
    
    disparities = model(
        # -- batch["img"][:, :, 0],
        # -- batch["img"][:, :, 1],
        snapshot_L,
        snapshot_R,
        iters=args.train_iters,
        test_mode=False,
    )
    num_traj = len(batch["disp"][0])
    for i in range(num_traj):
        seq_loss, metrics = sequence_loss(
            disparities[:, i], batch["disp"][:, i, 0], batch["valid_disp"][:, i, 0]
        )

        output[f"disp_{i}"] = {"loss": seq_loss / num_traj, "metrics": metrics}
    output["disparity"] = {
        "predictions": torch.cat(
            [disparities[-1, i, 0] for i in range(num_traj)], dim=1
        ).detach(),
    }
    return output


class Lite(LightningLite):
    def run(self, args):
        self.seed_everything(0)

        # ----------------------------------------- Loading Dataset -----------------------------------------------
        # -- Modified by Chu King on 15th November 2025 to allow quick testing with only 1 training video on the workstation.
        # -- The number of subframes should be fixed for SCI stereo.
        eval_dataloader_dr = datasets.DynamicReplicaDataset(
            # -- split="valid", sample_len=40, only_first_n_samples=1, VERBOSE=False
            split="valid", sample_len=args.sample_len, only_first_n_samples=1, VERBOSE=False
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
            "exp_dir": args.ckpt_path,
        }
        eval_vis_cfg = DefaultMunch.fromDict(eval_vis_cfg, object())
        evaluator.setup_visualization(eval_vis_cfg)

        # ----------------------------------------- Model Instantiation -----------------------------------------------
        # -- Added by Chu King on 20th November 2025
        # -- Instantiate the model
        model = wrapper(sigma_range=[0, 1e-9],
                        num_frames=args.sample_len,
                        in_channels=1,
                        n_taps=2,
                        resolution=args.image_size,
                        mixed_precision=args.mixed_precision,
                        attention_type=args.attention_type,
                        update_block_3d=args.update_block_3d,
                        different_update_blocks=args.different_update_blocks,
                        train_iters=args.train_iters)
        
        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        model.cuda()

        logging.info("count_parameters(model): {}".format(count_parameters(model)))

        train_loader = datasets.fetch_dataloader(args)
        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)

        logging.info(f"Train loader size:  {len(train_loader)}")

        optimizer, scheduler = fetch_optimizer(args, model)

        total_steps = 0
        logger = Logger(model, scheduler, args.ckpt_path)

        # ----------------------------------------- Loading Checkpoint -----------------------------------------------
        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")

        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(
                ".pt"
            )
            logging.info("Loading checkpoint...")
            strict = True

            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            # -- Since we wrapped the model in torch.nn.DataParallel or torch.nn.parallel.DistributedDataParallel,
            #    PyTorch automatically prefixes all parameter names with "module.":
            #        state_dict = {
            #            'module.conv1.weight': tensor(...),
            #            'module.conv1.bias': tensor(...),
            #            'module.fc.weight': tensor(...),
            #            'module.fc.bias': tensor(...),
            #        }
            # -- So we need to strip the "module." prefix: 
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=strict)

            logging.info(f"Done loading checkpoint")
        # ----------------------------------------- Optimzer, Scheduler -----------------------------------------------

        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        model.cuda()
        model.train()
        model.module.module.stereo.freeze_bn() # -- We keep BatchNorm frozen

        save_freq = args.save_freq
        scaler = GradScaler(enabled=args.mixed_precision)

        # ----------------------------------------- Training Loop -----------------------------------------------
        should_keep_training = True
        global_batch_num = 0
        epoch = -1
        while should_keep_training:
            epoch += 1

            for i_batch, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                if batch is None:
                    print("batch is None")
                    continue

                for k, v in batch.items():
                    batch[k] = v.cuda()

                assert model.training

                # ---- ---- FORWARD PASS ---- ----
                # -- Modified by Chu King on 20th November 2025
                output = model(batch)

                loss = 0
                logger.update()
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]
                        logger.writer.add_scalar(
                            f"live_{k}_loss", v["loss"].item(), total_steps
                        )
                    if "metrics" in v:
                        logger.push(v["metrics"], k)

                if self.global_rank == 0:
                    if total_steps % save_freq == save_freq - 1:
                        save_ims_to_tb(logger.writer, batch, output, total_steps)
                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    global_batch_num += 1
                self.barrier()

                # ---- ---- BACKWARD PASS ---- ----
                self.backward(scaler.scale(loss))
                scaler.unscale_(optimizer)

                # -- Prevent exploding gradients in RNNs or very deep networks
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                total_steps += 1

                if self.global_rank == 0:

                    if (i_batch >= len(train_loader) - 1) or (
                        total_steps == 1 and args.validate_at_start
                    ):
                        ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
                        save_path = Path(
                            f"{args.ckpt_path}/model_{args.name}_{ckpt_iter}.pth"
                        )

                        save_dict = {
                            "model": model.module.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "total_steps": total_steps,
                        }

                        logging.info(f"Saving file {save_path}")
                        self.save(save_dict, save_path)

                        # ---- ---- EVALUATION ---- ----
                        if epoch % args.evaluate_every_n_epoch == 0:
                            # -- Added by Chu King on 21st November 2025
                            model.eval()

                            logging.info(f"Evaluation at epoch {epoch}")
                            run_test_eval(
                                args.ckpt_path,
                                "valid",
                                evaluator,
                                model.module.module.sci_enc_L,
                                model.module.module.sci_enc_R,
                                model.module.module.stereo,
                                eval_dataloaders,
                                logger.writer,
                                total_steps,
                                resolution=args.image_size
                            )

                            # -- Added by Chu King on 20th November 2025 for SCI stereo
                            model.train()

                            model.module.module.stereo.freeze_bn()

                self.barrier()
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        # ----------------------------------------- Save models after training -----------------------------------------------
        # -- Modified by Chu King on 20th November 2025 to save SCI encoders' models.
        # -- PATH = f"{args.ckpt_path}/{args.name}_final.pth"
        PATH = f"{args.ckpt_path}/{args.name}_model_final.pth"
        torch.save(model.module.module.state_dict(), PATH)

        # ----------------------------------------- Testing -----------------------------------------------
        # -- Modified by Chu King on 20th November 2025
        test_dataloader_dr = datasets.DynamicStereoDataset(
            # -- The number of subframes should be fixed for SCI stereo
            # -- split="test", sample_len=150, only_first_n_samples=1
            split="test", sample_len=args.sample_len, only_first_n_samples=1
        )
        test_dataloaders = [
            ("sintel_clean", eval_dataloader_sintel_clean),
            ("sintel_final", eval_dataloader_sintel_final),
            ("dynamic_replica", test_dataloader_dr),
        ]

        # -- Modifed by Chu King on 21st November 2025
        model.eval()
        run_test_eval(
            args.ckpt_path,
            "test",
            evaluator,
            model.module.module.sci_enc_L,
            model.module.module.sci_enc_R,
            model.module.module.stereo,
            test_dataloaders,
            logger.writer,
            total_steps,
            resolution=args.image_size
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="dynamic-stereo", help="name your experiment")
    parser.add_argument("--restore_ckpt", help="restore checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=6, help="batch size used during training."
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        default=["things", "monkaa", "driving"],
        help="training datasets.",
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="max learning rate.")

    parser.add_argument(
        "--num_steps", type=int, default=100000, help="length of training schedule."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=[320, 720],
        help="size of the random image crops used during training.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=16,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )

    parser.add_argument(
        "--sample_len", type=int, default=2, help="length of training video samples"
    )
    parser.add_argument(
        "--validate_at_start", action="store_true", help="validate the model at start"
    )
    parser.add_argument("--save_freq", type=int, default=100, help="save frequency")

    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate every n epoch",
    )

    parser.add_argument(
        "--num_workers", type=int, default=6, help="number of dataloader workers."
    )
    # Validation parameters
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of updates to the disparity field in each forward pass during validation.",
    )
    # Architecure choices
    parser.add_argument(
        "--different_update_blocks",
        action="store_true",
        help="use different update blocks for each resolution",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        help="attention type of the SST and update blocks. \
            Any combination of 'self_stereo', 'temporal', 'update_time', 'update_space' connected by an underscore.",
    )
    parser.add_argument(
        "--update_block_3d", action="store_true", help="use Conv3D update block"
    )
    # Data augmentation
    parser.add_argument(
        "--img_gamma", type=float, nargs="+", default=None, help="gamma range"
    )
    parser.add_argument(
        "--saturation_range",
        type=float,
        nargs="+",
        default=None,
        help="color saturation",
    )
    parser.add_argument(
        "--do_flip",
        default=False,
        choices=["h", "v"],
        help="flip the images horizontally or vertically",
    )
    parser.add_argument(
        "--spatial_scale",
        type=float,
        nargs="+",
        default=[0, 0],
        help="re-scale the images randomly",
    )
    parser.add_argument(
        "--noyjitter",
        action="store_true",
        help="don't simulate imperfect rectification",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        accelerator="gpu",
        precision=32,
    ).run(args)
