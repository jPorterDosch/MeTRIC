# --------------------------------------------------------
# MeTRIC: fine-tune depth-conditioned StreamVGGT (LoRA + DepthConditioner).
# Head-injection (control) and token-injection (proposed) launch from this
# same entrypoint with only depth_cond.injection changed; the confound rule
# is enforced via the experiment manifest.
# Adapted from finetune.py (CUT3R/DUSt3R training code).
# --------------------------------------------------------
import datetime
import json
import math
import os
import pathlib
import random
import sys
import time
from pathlib import Path
from typing import Sized

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from streamvggt.loss.loss import *  # noqa: F401,F403 needed to eval() the criterion strings
from dust3r.inference import loss_of_one_batch  # noqa
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import DictConfig, OmegaConf

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

from streamvggt.depth_cond import (
    MetricCfg,
    MetricStreamVGGT,
    assert_confound_rule,
    build_metric_cfg,
    experiment_manifest,
    manifest_comparable_hash,
    simulate_sparse_depth,
)
from finetune import save_current_code, setup_for_distributed, build_dataset  # reuse

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

WANDB_PROJECT = "MeTRIC"
WANDB_ENTITY = "jporterdosch-university-of-tennessee-knoxville"


# ---------------------------------------------------------------------------
# model + manifest
# ---------------------------------------------------------------------------


def build_model(
    args: DictConfig,
    mcfg: MetricCfg,
    device: torch.device,
    load_pretrained: bool = True,
) -> tuple[MetricStreamVGGT, dict]:
    model = MetricStreamVGGT(mcfg)
    if load_pretrained and args.pretrained and not args.resume:
        print(f"Loading pretrained StreamVGGT: {args.pretrained}")
        print(model.load_pretrained(args.pretrained))
    n = model.apply_lora_adapters()
    stats = model.freeze_for_finetune()
    model.to(device)
    print(
        f"LoRA: wrapped {n} attention modules (targets={mcfg.lora.targets}, rank={mcfg.lora.rank})"
    )
    print(
        f"Params: total {stats['total_params']:,} | trainable {stats['trainable_params']:,} "
        f"({stats['trainable_pct']:.3f}%) | base attention frozen: {stats['base_attention_frozen']}"
    )
    if not stats["base_attention_frozen"]:
        raise RuntimeError("base attention projections must stay frozen")

    return model, stats


def write_manifest_and_check(args: DictConfig, mcfg: MetricCfg) -> dict:
    manifest = experiment_manifest(mcfg)
    manifest["_comparable_hash"] = manifest_comparable_hash(mcfg)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(
        f"Experiment manifest written to {path} (comparable hash {manifest['_comparable_hash'][:12]})"
    )
    other = (
        args.get("compare_with_manifest", None)
        if hasattr(args, "get")
        else getattr(args, "compare_with_manifest", None)
    )
    if other:
        with open(other) as f:
            other_manifest = json.load(f)
        other_manifest.pop("_comparable_hash", None)
        assert_confound_rule(other_manifest, experiment_manifest(mcfg))
        print(
            f"Confound rule OK against {other}: runs differ only in depth_cond.injection"
        )
    return manifest


# ---------------------------------------------------------------------------
# smoke mode: 5-step overfit on one synthetic clip, real criterion, no datasets
# ---------------------------------------------------------------------------


def make_synthetic_clip(
    num_views: int,
    B: int = 1,
    H: int = 154,
    W: int = 140,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> list[dict]:
    """A geometrically consistent synthetic clip: smooth RGB, metric depth,
    identity poses, pinhole intrinsics, pts3d by unprojection."""
    g = torch.Generator().manual_seed(seed)
    f = 0.8 * max(H, W)
    K = torch.tensor([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]])
    us = torch.arange(W).float().unsqueeze(0).expand(H, W)
    vs = torch.arange(H).float().unsqueeze(1).expand(H, W)
    ys = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    views = []
    for s in range(num_views):
        img = torch.zeros(B, 3, H, W)
        img[:, 0] = ys
        img[:, 1] = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
        img[:, 2] = 0.5
        img = (img + 0.1 * torch.rand(B, 3, H, W, generator=g)).clamp(0, 1)
        depth = 1.5 + 3.0 * ys + 0.2 * torch.rand(B, H, W, generator=g) + 0.05 * s
        z = depth
        pts3d = torch.stack(
            [(us - W / 2) / f * z, (vs - H / 2) / f * z, z], dim=-1
        )  # [B,H,W,3], camera frame == world frame (identity pose)
        view = {
            "img": img * 2 - 1,  # dataset convention [-1,1]; train loop maps to [0,1]
            "depthmap": depth,
            "pts3d": pts3d,
            "valid_mask": torch.ones(B, H, W, dtype=torch.bool),
            "sky_mask": torch.zeros(B, H, W, dtype=torch.bool),
            "camera_pose": torch.eye(4).unsqueeze(0).expand(B, 4, 4).contiguous(),
            "camera_intrinsics": K.unsqueeze(0).expand(B, 3, 3).contiguous(),
            "camera_only": torch.zeros(B, dtype=torch.bool),
            "is_metric": torch.ones(B, dtype=torch.bool),
            "is_metric_scale": torch.ones(B, dtype=torch.bool),
        }
        views.append({k: v.to(device) for k, v in view.items()})
    return views


def run_smoke(args: DictConfig, mcfg: MetricCfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, _ = build_model(args, mcfg, device)
    model.train()

    criterion = eval(args.train_criterion).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.smoke_lr), betas=(0.9, 0.95))

    batch = make_synthetic_clip(
        int(args.smoke_num_views), device=device, seed=args.seed
    )
    for view in batch:
        view["img"] = (view["img"] + 1.0) / 2.0  # same range mapping as the train loop
    simulate_sparse_depth(batch, mcfg.depth_cond.sim_num_points)

    losses = []
    for step in range(int(args.smoke_steps)):
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator=None,
            symmetrize_batch=False,
            use_amp=True,
        )
        loss, loss_details = result["loss"]
        if not math.isfinite(float(loss)):
            print(f"Loss is {float(loss)}, details: {loss_details}")
            sys.exit(1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss))
        print(
            f"[smoke][{mcfg.depth_cond.injection}] step {step}: loss = {float(loss):.6f}"
        )

    out = {
        "injection": mcfg.depth_cond.injection,
        "losses": losses,
        "grad_checkpoint": mcfg.train.grad_checkpoint,
    }
    with open(os.path.join(args.output_dir, "smoke_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    decreased = losses[-1] < losses[0]
    rises = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
    print(
        f"[smoke] loss {losses[0]:.6f} -> {losses[-1]:.6f} (rises: {rises}/{len(losses) - 1})"
    )
    if not decreased:
        print("[smoke] FAIL: loss did not decrease")
        sys.exit(1)
    print("[smoke] PASS")


# ---------------------------------------------------------------------------
# full training (accelerate), mirrors finetune.py with depth conditioning
# ---------------------------------------------------------------------------


def train(args: DictConfig, mcfg: MetricCfg) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        log_with="wandb",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device
    setup_for_distributed(accelerator)

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    wandb_init_kwargs = {"name": args.exp_name, "dir": args.output_dir}
    if WANDB_ENTITY:
        wandb_init_kwargs["entity"] = WANDB_ENTITY
    accelerator.init_trackers(
        project_name=WANDB_PROJECT,
        config=OmegaConf.to_container(args, resolve=True),
        init_kwargs={"wandb": wandb_init_kwargs},
    )

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    printer.info("Building train dataset %s", args.train_dataset)
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length,
    )

    printer.info("Loading depth-conditioned model")
    model, _ = build_model(args, mcfg, device)

    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)

    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(accelerator=accelerator)

    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")

    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )

    def save_model(
        epoch: int, fname: str, best_so_far: float, data_iter_step: int
    ) -> None:
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=accelerator.unwrap_model(model),
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far, args.start_step)
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far, args.start_step)
        if epoch >= args.epochs:
            break

        train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            args=args,
            mcfg=mcfg,
        )

    total_time = time.time() - start_time
    printer.info(
        "Training time {}".format(str(datetime.timedelta(seconds=int(total_time))))
    )

    output_dir = Path(args.output_dir)
    to_save = {
        "args": args,
        "model": accelerator.unwrap_model(model).cpu().state_dict(),
        "epoch": args.epochs,
    }
    printer.info(f">> Saving model to {output_dir / 'checkpoint-final.pth'} ...")
    misc.save_on_master(accelerator, to_save, output_dir / "checkpoint-final.pth")
    accelerator.end_training()


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler: NativeScaler,
    args: DictConfig,
    mcfg: MetricCfg,
) -> dict:
    if not torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("TF32 matmul must stay enabled (set at module import)")

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        with accelerator.accumulate(model):
            if isinstance(batch, list) and all(
                isinstance(v, dict) and "img" in v for v in batch
            ):
                for view in batch:
                    view["img"] = (view["img"] + 1.0) / 2.0

            # depth conditioning input: sparse metric samples of the GT depth
            simulate_sparse_depth(batch, mcfg.depth_cond.sim_num_points)

            epoch_f = epoch + data_iter_step / len(data_loader)
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)
            step = int(epoch_f * len(data_loader))

            result = loss_of_one_batch(
                batch,
                model,
                criterion,
                accelerator,
                inference=False,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
            )
            loss, loss_details = result["loss"]
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
                sys.exit(1)
            if not result.get("already_backprop", False):
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

            del loss, batch

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)
            metric_logger.update(loss=loss_value, **loss_details)

            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()
                log_dict = {
                    "train_loss": loss_value_reduce,
                    "train_lr": lr,
                    "epoch": epoch_f,
                }
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor) and val.ndim > 0:
                        continue
                    if isinstance(val, dict):
                        continue
                    log_dict["train_" + name] = val
                accelerator.log(misc.aggregate_per_view_metrics(log_dict), step=step)

    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="finetune_depth.yaml",
)
def run(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    mcfg = build_metric_cfg(cfg)
    write_manifest_and_check(cfg, mcfg)

    if cfg.smoke_test:
        run_smoke(cfg, mcfg)
    else:
        train(cfg, mcfg)


if __name__ == "__main__":
    run()
