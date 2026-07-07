# --------------------------------------------------------
# MeTRIC: fine-tune depth-conditioned StreamVGGT (LoRA + DepthConditioner).
# Head-injection (control) and token-injection (proposed) launch from this
# same entrypoint with only --depth-cond.injection changed. Each run is named
# by a canonical SHA over its config; directory collisions fail fast so a
# finished experiment is never silently re-run, and the manifest is logged to
# wandb where runs can be filtered for comparison.
# Adapted from finetune.py (CUT3R/DUSt3R training code).
# --------------------------------------------------------
import argparse
import dataclasses
import datetime
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from collections.abc import Callable
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

import tyro

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

from streamvggt.depth_cond import (
    DepthCondCfg,
    EncoderCacheCfg,
    LoRACfg,
    MetricCfg,
    MetricStreamVGGT,
    TrainCondCfg,
    experiment_hash,
    experiment_manifest,
    seed_everything,
    simulate_sparse_depth,
)
from finetune import save_current_code, setup_for_distributed, build_dataset  # reuse

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

WANDB_PROJECT = "MeTRIC"
WANDB_ENTITY = "jporterdosch-university-of-tennessee-knoxville"

_DEFAULT_RESOLUTIONS = (
    "[(518, 392), (518, 336), (518, 294), (518, 266), (518, 210), (518, 154), "
    "(392, 518), (336, 518), (294, 518), (266, 518)]"
)


@dataclass
class FinetuneDepthCfg:
    """Depth-conditioned StreamVGGT fine-tuning. All depth-conditioning /
    LoRA / cache knobs live in the nested dataclasses (tyro exposes them as
    e.g. --depth-cond.injection, --lora.rank)."""

    depth_cond: DepthCondCfg = field(default_factory=DepthCondCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    encoder_cache: EncoderCacheCfg = field(default_factory=EncoderCacheCfg)
    train: TrainCondCfg = field(default_factory=TrainCondCfg)

    # checkpointing / identity
    pretrained: str = "../ckpt/checkpoints.pth"
    resume: str | None = None
    save_dir: str = "../checkpoints"
    exp_name: str = "metric_depth_cond"

    # optimization
    seed: int = 0
    batch_size: int = 1
    accum_iter: int = 1
    epochs: int = 10
    start_epoch: int = 0
    start_step: int = 0
    weight_decay: float = 0.05
    lr: float = 1e-5
    min_lr: float = 1e-7
    warmup_epochs: float = 0.5
    amp: int = 1

    # data
    num_views: int = 10
    num_workers: int = 12
    fixed_length: bool = True
    benchmark: bool = False
    train_dataset: str = (
        "4500 @ ARKitScenes_Multi(allow_repeat=False, split='train', "
        "ROOT='../data/train/processed_arkitscenes/', aug_crop=16, "
        f"resolution={_DEFAULT_RESOLUTIONS}, transform=SeqColorJitter, "
        "num_views=10, n_corres=0)"
    )
    train_criterion: str = (
        "ConfLoss(Regr3DPose(L21, norm_mode='?avg_dis'), alpha=0.2) + FinetuneLoss()"
    )

    # logging / saving cadence (not part of the experiment identity)
    print_freq: int = 10
    save_freq: float = 0.1
    keep_freq: int = 1

    # derived at startup (do not set on the CLI)
    output_dir: str = ""


# Blacklist: the only fields that do NOT define the experiment (output naming,
# logging cadence, worker counts, resume bookkeeping). Everything else in the
# config -- nested depth_cond/lora/cache/train blocks included -- is hashed.
_NON_IDENTITY_FIELDS = (
    "save_dir",
    "exp_name",
    "output_dir",
    "resume",
    "start_epoch",
    "start_step",
    "print_freq",
    "save_freq",
    "keep_freq",
    "num_workers",
    "benchmark",
)


def build_manifest(cfg: FinetuneDepthCfg) -> dict:
    return experiment_manifest(cfg, exclude=_NON_IDENTITY_FIELDS)


def _is_rank_zero() -> bool:
    """True on the main process, including before Accelerator/dist init exists
    (torchrun/accelerate launch export RANK / LOCAL_RANK to every process)."""
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


def _picklable_args(cfg: FinetuneDepthCfg) -> argparse.Namespace:
    """Config snapshot safe to embed in checkpoints. FinetuneDepthCfg lives in
    __main__, so pickling the dataclass itself would make every checkpoint
    unloadable from any other script (eval scripts torch.load the whole file);
    a Namespace of plain dicts is importable everywhere and still offers the
    attribute access croco's misc.save_model needs (args.output_dir)."""
    return argparse.Namespace(**dataclasses.asdict(cfg))


def resolve_output_dir(cfg: FinetuneDepthCfg, run_hash: str) -> str:
    """Derive the save directory (<save_dir>/<exp_name>_<hash>) and fail fast
    if it already exists: an experiment with this exact config has been run or
    is running, and silently re-running it would waste the compute. To resume
    an interrupted run, pass --resume <path/to/checkpoint-last.pth> explicitly.

    Only rank 0 performs the existence check: under multi-process launch the
    non-zero ranks start later and would otherwise see the directory rank 0
    just created and abort the whole job."""
    output_dir = os.path.join(cfg.save_dir, f"{cfg.exp_name}_{run_hash[:10]}")
    if cfg.resume or not _is_rank_zero():
        return output_dir
    if os.path.exists(output_dir):
        raise RuntimeError(
            f"Output dir {output_dir} already exists: an experiment with this exact "
            "config hash has already been launched. Refusing to re-run. Either change "
            f"the config, pass --resume {os.path.join(output_dir, 'checkpoint-last.pth')} "
            "to continue an interrupted run, or remove the directory deliberately."
        )
    return output_dir


def build_model(
    args: FinetuneDepthCfg,
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


def train(
    args: FinetuneDepthCfg, mcfg: MetricCfg, manifest: dict, run_hash: str
) -> None:
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
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # the manifest goes to disk AND to wandb so runs can be filtered for
    # comparison (e.g. head-vs-token pairs agreeing on every other knob)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
            json.dump(
                {**manifest, "experiment_hash": run_hash}, f, indent=2, sort_keys=True
            )

    wandb_config = {**dataclasses.asdict(args), **manifest, "experiment_hash": run_hash}
    wandb_init_kwargs = {
        "name": f"{args.exp_name}_{run_hash[:10]}",
        "dir": args.output_dir,
    }
    if WANDB_ENTITY:
        wandb_init_kwargs["entity"] = WANDB_ENTITY
    accelerator.init_trackers(
        project_name=WANDB_PROJECT,
        config=wandb_config,
        init_kwargs={"wandb": wandb_init_kwargs},
    )

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    seed_everything(seed)
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
            args=_picklable_args(args),
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
        # For fractional start_epoch, this will fire every epoch: this is intentional, want to frequently save for long epochs in case of crash.
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
            save_model=save_model,
        )

    total_time = time.time() - start_time
    printer.info(
        "Training time {}".format(str(datetime.timedelta(seconds=int(total_time))))
    )

    output_dir = Path(args.output_dir)
    to_save = {
        "args": _picklable_args(args),
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
    args: FinetuneDepthCfg,
    mcfg: MetricCfg,
    save_model: Callable[[int, str, float, int], None] | None = None,
) -> dict:
    if not torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("TF32 matmul must stay enabled (set at module import)")

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    # duck-typed set_epoch plumbing ported verbatim from finetune.py
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

            # depth conditioning input: patch-masked samples of the GT depth
            simulate_sparse_depth(
                batch,
                mode=mcfg.depth_cond.sim_mode,
                patch_size=mcfg.depth_cond.sim_patch_size,
                mask_ratio=mcfg.depth_cond.sim_mask_ratio,
            )

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

        # mid-epoch checkpoint-last saves (ported from finetune.py): without
        # these, a crash/preemption inside a long epoch loses all progress and
        # leaves nothing for --resume to point at
        if save_model is not None:
            save_every = int(args.save_freq * len(data_loader))
            if (
                save_every > 0
                and data_iter_step % save_every == 0
                and data_iter_step != 0
                and data_iter_step != len(data_loader) - 1
            ):
                print("saving at step", data_iter_step)
                save_model(epoch - 1, "last", float("inf"), data_iter_step)

    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(cfg: FinetuneDepthCfg) -> None:
    mcfg = MetricCfg(
        depth_cond=cfg.depth_cond,
        lora=cfg.lora,
        encoder_cache=cfg.encoder_cache,
        train=cfg.train,
    ).validate()

    manifest = build_manifest(cfg)
    run_hash = experiment_hash(manifest)
    cfg.output_dir = resolve_output_dir(cfg, run_hash)
    print(f"Experiment {cfg.exp_name} hash {run_hash[:10]} -> {cfg.output_dir}")

    train(cfg, mcfg, manifest, run_hash)


if __name__ == "__main__":
    main(tyro.cli(FinetuneDepthCfg))
