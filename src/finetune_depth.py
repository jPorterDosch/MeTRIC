# --------------------------------------------------------
# MeTRIC: fine-tune depth-conditioned StreamVGGT (LoRA + DepthConditioner).
# Head-injection (control) and token-injection (proposed) launch from this
# same entrypoint with only --depth-cond.injection changed. Each run is named
# by a canonical SHA over its config; directory collisions fail fast so a
# finished experiment is never silently re-run, and the manifest is logged to
# wandb where runs can be filtered for comparison.
# Adapted from finetune.py (CUT3R/DUSt3R training code).
# --------------------------------------------------------
import datetime
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from streamvggt.loss import LossConfig
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
    experiment_id,
    experiment_manifest,
    seed_everything,
    simulate_sparse_depth,
)
from eval.temporal_consistency.metrics import depth_evaluation, tae
from finetune import save_current_code, setup_for_distributed  # reuse
from streamvggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb
from streamvggt.datasets import (
    CatDataset,
    DatasetName,
    MultiDatasetConfig,
    Split,
    TransformName,
    get_data_loader,
)
from train_utils import picklable_args, resolve_output_dir, to_primitive

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

WANDB_PROJECT = "MeTRIC"
WANDB_ENTITY = "sparse_representation_learning"

_ARKITSCENES_RESOLUTIONS = (
    (518, 392),
    (518, 336),
    (518, 294),
    (518, 266),
    (518, 210),
    (518, 154),
    (392, 518),
    (336, 518),
    (294, 518),
    (266, 518),
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
    exp_group: str = "metric_depth_cond"
    """Experiment / sweep label. Every run sharing it is bucketed under one group
    in the wandb UI and nests its runs under ``<save_dir>/<exp_group>/<run_id>``; the
    run_id (config hash) distinguishes the individual arms within the sweep, so a
    whole ablation ladder gets one --exp-group. Naming only -- NOT part of the
    config hash."""

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

    # data -- the datasets are a nested config of parallel per-dataset tuples
    # (tyro exposes --dataset.root A B, --dataset.dataset arkitscenes_lowres
    # arkitscenes_highres, --dataset.epoch-size 4500 2250, ...); entry i of
    # every tuple describes dataset i and lengths are validated up front. It is
    # the single source of truth for num_views/resolution (shared across the
    # mixture); build with dataset.build_all() and concatenate -- no eval.
    # The default reproduces the original recipe's ARKitScenes slice:
    # 4500 @ lowres + 2250 @ highres (the loaders partition the scenes).
    train_dataset: MultiDatasetConfig = field(
        default_factory=lambda: MultiDatasetConfig(
            root=(
                Path("../data/train/processed_arkitscenes/"),
                Path("../data/train/processed_arkitscenes_highres/"),
            ),
            dataset=(
                DatasetName.ARKITSCENES_LOWRES,
                DatasetName.ARKITSCENES_HIGHRES,
            ),
            max_interval=(8, 8),
            epoch_size=(4500, 2250),
            # the lowres loader excludes the highres tree's scenes; pass the
            # real root explicitly so the partition cannot silently break if
            # the roots stop following the <x>/<x>_highres naming convention
            highres_root=(
                Path("../data/train/processed_arkitscenes_highres/"),
                None,
            ),
            num_views=10,
            resolution=_ARKITSCENES_RESOLUTIONS,
            split=Split.TRAIN,
            aug_crop=16,
            transform=TransformName.SEQ_COLOR_JITTER,
            n_corres=0,
        )
    )

    # The default mirrors the original recipe's test slice:
    # 1000 @ ARKitScenes lowres, split test, single (518, 392) resolution,
    # num_views 4, seed 42 (per-__getitem__ reseed -> deterministic clips).
    # Note Split.VAL does not exist -- the dataset classes only accept
    # TRAIN/TEST (highres maps TEST to its Validation/ tree on disk).
    val_dataset: MultiDatasetConfig = field(
        default_factory=lambda: MultiDatasetConfig(
            root=(Path("../data/train/processed_arkitscenes/"),),
            dataset=(DatasetName.ARKITSCENES_LOWRES,),
            max_interval=(8,),
            epoch_size=(1000,),
            highres_root=(Path("../data/train/processed_arkitscenes_highres/"),),
            num_views=4,
            resolution=((518, 392),),
            split=Split.TEST,
            n_corres=0,
            seed=42,
        )
    )

    num_workers: int = 12
    fixed_length: bool = True
    benchmark: bool = False

    # Training objective. tyro exposes the knobs as e.g. --loss.recipe,
    # --loss.pixel-loss, --loss.conf-alpha. The default recipe is DEPTH_TRAIN
    # (DepthTrainLoss: temporal depth-gradient consistency + depth accuracy);
    # pass --loss.recipe finetune_train for the historical
    #   ConfLoss(Regr3DPose(L21, norm_mode='?avg_dis'), alpha=0.2) + FinetuneLoss()
    loss: LossConfig = field(default_factory=LossConfig)

    # Validation configuration
    val_freq: int = 1

    # logging / saving cadence (not part of the experiment identity)
    print_freq: int = 10
    save_freq: float = 0.1

    # visualization: export predicted-depth point clouds (.glb) during the
    # val / streaming-eval passes to <output_dir>/glb/. Off by default; the
    # cap bounds how many clips are written per pass (accumulated over the
    # first batches). Not part of the experiment identity.
    export_glb: bool = False
    export_glb_max_clips: int = 4

    # derived at startup (do not set on the CLI)
    output_dir: str = ""


# Blacklist: the only fields that do NOT define the experiment (output naming,
# logging cadence, worker counts, resume bookkeeping). Everything else in the
# config -- nested depth_cond/lora/cache/train blocks included -- is hashed.
_NON_IDENTITY_FIELDS = (
    "save_dir",
    "exp_group",
    "output_dir",
    "resume",
    "start_epoch",
    "start_step",
    "print_freq",
    "save_freq",
    "num_workers",
    "benchmark",
    "export_glb",
    "export_glb_max_clips",
)


def build_manifest(cfg: FinetuneDepthCfg) -> dict:
    return experiment_manifest(cfg, exclude=_NON_IDENTITY_FIELDS)


def _dataset_tag(config: MultiDatasetConfig) -> str:
    """Short wandb section tag for a dataset mixture: "hammer",
    "hammer+scannet", ... Robust to the enum members not being coerced yet
    (plain strings before validate())."""
    return "+".join(getattr(d, "value", str(d)) for d in config.dataset)


def build_train_loader(
    args: FinetuneDepthCfg,
    split: Split,
    accelerator,
    batch_size: int | None = None,
) -> torch.utils.data.DataLoader:
    """Build the dataset mixture (one CatDataset over every configured
    dataset, mirroring the original `N @ ds1 + M @ ds2` recipes) and wrap it
    in the batched-sampler loader. batch_size overrides args.batch_size when
    given (streaming_eval needs a batch-1 loader over the val config).
    """
    if batch_size is None:
        batch_size = args.batch_size
    match split:
        case Split.TRAIN:
            printer.info("Building train datasets %s", args.train_dataset)
            train_datasets = args.train_dataset.build_all()
            train_dataset = (
                train_datasets[0]
                if len(train_datasets) == 1
                else CatDataset(train_datasets)
            )
            return get_data_loader(
                train_dataset,
                batch_size=batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                drop_last=True,
                accelerator=accelerator,
                fixed_length=args.fixed_length,
            )
        case Split.TEST:
            printer.info("Building validation datasets %s", args.val_dataset)
            val_datasets = args.val_dataset.build_all()
            val_dataset = (
                val_datasets[0] if len(val_datasets) == 1 else CatDataset(val_datasets)
            )
            # shuffle=True is still deterministic here: every draw in
            # CustomRandomSampler comes from a rng seeded by the epoch number
            # (epoch + 788), and ResizedDataset's 1000-slot mapping from
            # epoch + 777 -- val_loop pins both with set_epoch(0) each pass.
            # (make_sampler raises NotImplementedError on shuffle=False.)
            return get_data_loader(
                val_dataset,
                batch_size=batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                drop_last=False,
                accelerator=accelerator,
                fixed_length=args.fixed_length,
            )

        case _:
            raise ValueError(f"Expected split in {list(Split)}, got: {split}.")


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


def run(
    args: FinetuneDepthCfg, mcfg: MetricCfg, manifest: dict, run_hash: str, run_id: str
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
    record = {"experiment_hash": run_hash, "experiment_id": run_id}
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
            json.dump({**manifest, **record}, f, indent=2, sort_keys=True)

    wandb_config = {**to_primitive(args), **manifest, **record}
    wandb_init_kwargs = {
        "name": f"{args.exp_group}_{run_id}",
        # group -> every run sharing exp_group is bucketed together in the wandb
        # UI (e.g. all arms of one ablation ladder); run_id keeps names unique
        "group": args.exp_group,
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

    data_loader_train = build_train_loader(args, Split.TRAIN, accelerator)
    data_loader_val = (
        build_train_loader(args, Split.TEST, accelerator) if args.val_freq > 0 else None
    )
    # streaming_eval drives StreamVGGT.inference, which folds the batch dim of
    # frame["img"] into the sequence, so it must see one clip at a time: give
    # it its own batch-1 loader over the val config instead of constraining
    # the whole run's batch_size (reuse the val loader when it is already 1)
    data_loader_stream = None
    if data_loader_val is not None:
        data_loader_stream = (
            data_loader_val
            if args.batch_size == 1
            else build_train_loader(args, Split.TEST, accelerator, batch_size=1)
        )

    printer.info("Loading depth-conditioned model")
    model, _ = build_model(args, mcfg, device)

    train_criterion = args.loss.build().to(device)
    printer.info(f">> Creating train criterion = {train_criterion!r}")

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
    if data_loader_val is not None:
        stream_is_val = data_loader_stream is data_loader_val
        data_loader_val = accelerator.prepare(data_loader_val)
        # the stream loader is prepared too so it shards across ranks the same
        # way (_reduce_depth_metrics assumes every rank walked its own shard)
        data_loader_stream = (
            data_loader_val
            if stream_is_val
            else accelerator.prepare(data_loader_stream)
        )

    def save_model(
        epoch: int, fname: str, best_so_far: float, data_iter_step: int
    ) -> None:
        misc.save_model(
            accelerator=accelerator,
            args=picklable_args(args),
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
        if epoch >= args.epochs:
            break

        train_loop(
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
            best_so_far=best_so_far,
        )

        # the loop breaks at `epoch >= args.epochs` before training, so the
        # last epoch that reaches this point is args.epochs - 1
        is_last_epoch = epoch == args.epochs - 1
        if data_loader_val is not None and (
            epoch % args.val_freq == 0 or is_last_epoch
        ):
            val_stats = val_loop(
                model,
                train_criterion,
                data_loader_val,
                accelerator,
                epoch,
                # training already logged up to (epoch+1)*len this epoch, and
                # wandb drops rows whose step is below the current max
                step=(epoch + 1) * len(data_loader_train),
                args=args,
                mcfg=mcfg,
                prefix=f"val/{_dataset_tag(args.val_dataset)}",
            )
            # Select "best" on metric AbsRel, NOT the criterion loss. The
            # confidence-regularized loss (-alpha*log sigma) is not monotonic
            # with depth quality: it can keep falling as the confidence inflates
            # while AbsRel plateaus/worsens, so selecting on loss_avg pinned
            # "best" to the earliest epoch. absrel_metric_avg (metric-scale,
            # lower is better) tracks the actual objective and is already
            # globally reduced across ranks (_reduce_depth_metrics), so it needs
            # no median/avg caveat. Fall back to loss_avg only if the metric
            # never fired on any rank (e.g. a val pass with no valid GT pixels).
            selection = val_stats.get("absrel_metric_avg", val_stats["loss_avg"])
            if selection < best_so_far:
                best_so_far = selection
                save_model(epoch, "best", best_so_far, args.start_step)

    total_time = time.time() - start_time
    printer.info(
        "Training time {}".format(str(datetime.timedelta(seconds=int(total_time))))
    )

    # final causal evaluation on the deployment (per-frame KV-cache) path;
    # must run before the final save below moves the model to cpu
    if data_loader_stream is not None:
        streaming_eval(
            model,
            data_loader_stream,
            accelerator,
            step=args.epochs * len(data_loader_train),
            args=args,
            mcfg=mcfg,
            prefix=f"final_stream/{_dataset_tag(args.val_dataset)}",
        )

    output_dir = Path(args.output_dir)
    to_save = {
        "args": picklable_args(args),
        "model": accelerator.unwrap_model(model).cpu().state_dict(),
        "epoch": args.epochs,
    }
    printer.info(f">> Saving model to {output_dir / 'checkpoint-final.pth'} ...")
    misc.save_on_master(accelerator, to_save, output_dir / "checkpoint-final.pth")
    accelerator.end_training()


def _set_data_epoch(data_loader: DataLoader, epoch: int) -> None:
    """Propagate the epoch to the dataset and the sampler stack. Delegates to
    accelerate's DataLoaderShard.set_epoch, which handles both the
    single-process layout (batch_sampler IS the raw BatchedRandomSampler) and
    the sharded one (BatchSamplerShard wrapping it). The duck-typed
    data_loader.batch_sampler.batch_sampler hop this replaces silently no-oped
    on single-process runs, leaving the sampler epoch unset ('Epoch number not
    set' on the first batch). The dataset call stays explicit because
    DataLoaderShard.set_epoch skips it on the sharded layout (elif), and
    ResizedDataset.__getitem__ requires it; plain DataLoaders (tests) fall
    through every guard."""
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "set_epoch"):  # accelerate DataLoaderShard
        data_loader.set_epoch(epoch)
    elif hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):  # not accelerator.prepare'd: the raw streamvggt batched sampler
        data_loader.batch_sampler.set_epoch(epoch)


def _prepare_batch(batch: list[dict], mcfg: MetricCfg) -> None:
    """Shared train/val batch preprocessing, in this order: rescale imgs to
    [0, 1], then attach the sparse-depth conditioning input (patch-masked
    samples of the GT depth)."""
    if isinstance(batch, list) and all(
        isinstance(v, dict) and "img" in v for v in batch
    ):
        for view in batch:
            view["img"] = (view["img"] + 1.0) / 2.0
    simulate_sparse_depth(
        batch,
        mode=mcfg.depth_cond.sim_mode,
        patch_size=mcfg.depth_cond.sim_patch_size,
        mask_ratio=mcfg.depth_cond.sim_mask_ratio,
    )


def train_loop(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler: NativeScaler,
    args: FinetuneDepthCfg,
    mcfg: MetricCfg,
    save_model: Callable[[int, str, float, int], None] | None = None,
    best_so_far: float = float("inf"),
) -> dict:
    if not torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("TF32 matmul must stay enabled (set at module import)")

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    _set_data_epoch(data_loader, epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        with accelerator.accumulate(model):
            _prepare_batch(batch, mcfg)

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
                # "/"-namespaced keys so wandb groups metrics into sections
                # (train/..., val/<dataset>/..., stream/<dataset>/...)
                log_dict = {
                    "train/loss": loss_value_reduce,
                    "train/lr": lr,
                    "epoch": epoch_f,
                }
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor) and val.ndim > 0:
                        continue
                    if isinstance(val, dict):
                        continue
                    log_dict["train/" + name] = val
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
                # pass the live best_so_far: a hardcoded inf here poisoned
                # best tracking on resume from a mid-epoch checkpoint, letting
                # a worse model overwrite checkpoint-best.pth
                save_model(epoch - 1, "last", best_so_far, data_iter_step)

    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _stack_depth_batch(
    views: list[dict], preds: list[dict]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """views/preds are the [S]-lists loss_of_one_batch returns as
    result["views"]/result["pred"] -- the training forward and the streaming
    inference path emit the same per-view shapes. Real GT is g["depthmap"]
    (NOT g["depth"], the teacher output); p["depth"] is the prediction.
    Returns (pred, gt, valid, K, pose), all [B,S,...] on cpu."""
    pred = torch.stack([p["depth"].detach() for p in preds], dim=1)
    pred = pred.squeeze(-1).float().cpu()  # [B,S,H,W,1] -> [B,S,H,W]
    gt = torch.stack([g["depthmap"] for g in views], dim=1).float().cpu()
    valid = torch.stack([g["valid_mask"] for g in views], dim=1).cpu().bool()
    K = torch.stack([g["camera_intrinsics"] for g in views], dim=1).float().cpu()
    pose = torch.stack([g["camera_pose"] for g in views], dim=1).float().cpu()
    return pred, gt, valid, K, pose


def _clip_predictions(
    img: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    K: torch.Tensor,
    pose: torch.Tensor,
) -> dict:
    """Assemble the numpy `predictions` dict predictions_to_glb consumes, for a
    single clip. Inputs are the per-clip slices of _stack_depth_batch plus the
    images: img [S,3,H,W] in [0,1], depth [S,H,W], valid [S,H,W] bool,
    K [S,3,3], pose [S,4,4] cam2world.

    Both the unprojector and predictions_to_glb want world->cam extrinsics of
    shape [S,3,4] -- depth_to_world_coords_points is documented "cam from
    world", and the glb builder inverts extrinsic to place the camera frustums
    -- so we invert the cam2world pose ONCE and hand the same array to both;
    point cloud and cameras then live in one frame. Invalid pixels get zero
    confidence, which predictions_to_glb's conf>1e-5 filter drops (paired with
    conf_thres=0.0, so no valid pixels are thresholded out)."""
    world2cam = np.linalg.inv(pose.numpy())[:, :3, :4].astype(np.float32)  # [S,3,4]
    # the unprojector does a hard np.squeeze(-1) per frame, so it needs the
    # trailing singleton (_stack_depth_batch already dropped it -> [S,H,W])
    world_points = unproject_depth_map_to_point_map(
        depth.numpy()[..., None], world2cam, K.numpy()
    )  # [S,H,W,3]
    # confidence = GT-valid AND finite prediction. Dropping non-finite pixels
    # matters: a NaN/Inf predicted depth (bad/early ckpt) at a GT-valid pixel
    # would survive predictions_to_glb's conf>1e-5 filter and poison the
    # np.percentile scene-scale (-> NaN camera sizing for the whole clip).
    conf = (valid & torch.isfinite(depth)).numpy().astype(np.float32)
    return {
        "world_points_from_depth": world_points,
        "depth_conf": conf,
        "images": img.numpy(),
        "extrinsic": world2cam,
    }


def _export_eval_glbs(
    views: list[dict],
    preds: list[dict],
    out_dir: str,
    tag: str,
    accelerator: Accelerator,
    start_idx: int,
    max_clips: int,
) -> int:
    """Write up to (max_clips - start_idx) predicted-depth point clouds from
    THIS batch to <out_dir>/glb/<tag>_clipN.glb and return how many were
    written, so the caller can accumulate across batches until it reaches
    max_clips. N is a pass-global index so clips from successive batches do not
    collide. Main process only: the export is redundant across ranks (the val /
    stream loaders shard the same clip set), and predictions_to_glb prints /
    does file IO we do not want N-fold."""
    if not accelerator.is_main_process or start_idx >= max_clips:
        return 0
    glb_dir = os.path.join(out_dir, "glb")
    os.makedirs(glb_dir, exist_ok=True)
    # img is not in _stack_depth_batch; stack it the same way ([B,S,3,H,W])
    imgs = torch.stack([v["img"] for v in views], dim=1).float().cpu()
    pred, _, valid, K, pose = _stack_depth_batch(views, preds)
    written = 0
    for b in range(pred.shape[0]):
        if start_idx + written >= max_clips:
            break
        predictions = _clip_predictions(imgs[b], pred[b], valid[b], K[b], pose[b])
        # prediction_mode without "Pointmap" takes the world_points_from_depth
        # branch directly (no missing-key fallback warning)
        scene = predictions_to_glb(
            predictions,
            conf_thres=0.0,
            show_cam=True,
            prediction_mode="Depthmap and Camera",
        )
        scene.export(
            file_obj=os.path.join(glb_dir, f"{tag}_clip{start_idx + written}.glb")
        )
        written += 1
    return written


def _img2lidar(K: torch.Tensor, pose: torch.Tensor) -> np.ndarray:
    """Matrix mapping homogeneous pixel coords (u*z, v*z, z, 1) to world, the
    convention metrics.tae expects: inv(K_4x4 @ world2cam) = cam2world @
    inv(K_4x4). K [3,3], pose [4,4] cam2world. float32 throughout -- the
    inputs are float32 and float64 doubles the TAE reprojection cost."""
    k4 = np.eye(4, dtype=np.float32)
    k4[:3, :3] = K.numpy()
    return pose.numpy() @ np.linalg.inv(k4)


def _clip_dataset(views: list[dict], b: int) -> str:
    """Dataset label of clip b (all views in a clip come from one scene, hence
    one dataset). The collated 'dataset' field is a per-sample list/tuple; a
    bare str covers the unbatched case. Lowercased so keys are consistent
    regardless of loader casing (HAMMER_Multi emits 'hammer', ScanNet_Multi
    'ScanNet')."""
    d = views[0].get("dataset", "unknown")
    label = d if isinstance(d, str) else d[b]
    return str(label).lower()


def _val_depth_metrics(views: list[dict], preds: list[dict]) -> dict[str, list[float]]:
    """Sequence-level depth metrics for one batch: affine-invariant (one lstsq
    scale/shift over the whole clip) and metric (no-alignment) AbsRel /
    delta<1.25, plus TAE over adjacent frames of the ALIGNED prediction. The
    whole-clip alignment is non-causal -- fine for the training-forward val
    pass; the streaming eval uses _streaming_depth_metrics instead.
    Returns PER-CLIP value lists (not batch means) so the caller's
    accumulation weights every clip equally regardless of batch size. Keys are
    '<dataset>/<metric>' so a mixed-dataset val set yields per-dataset numbers
    (the caller blends them for the overall figure)."""
    pred, gt, valid, K, pose = _stack_depth_batch(views, preds)

    out: dict[str, list[float]] = {}
    B, S, H, W = gt.shape
    for b in range(B):
        mask = valid[b] & (gt[b] > 0)  # [S,H,W]
        if not mask.any():
            continue
        ds = _clip_dataset(views, b)
        # the 3rd return is the aligned full-size prediction, reused for TAE
        res_affine, _, aligned, _ = depth_evaluation(
            pred[b], gt[b], custom_mask=mask, scale_and_shift=True
        )
        # affine-variant: raw prediction vs GT, measures metric alignment
        res_metric, _, _, _ = depth_evaluation(
            pred[b], gt[b], custom_mask=mask, metric_scale=True
        )
        out.setdefault(f"{ds}/absrel_affine", []).append(res_affine["Abs Rel"])
        out.setdefault(f"{ds}/delta1_affine", []).append(res_affine["delta < 1.25"])
        out.setdefault(f"{ds}/rmse_affine", []).append(res_affine["RMSE"])
        out.setdefault(f"{ds}/absrel_metric", []).append(res_metric["Abs Rel"])
        out.setdefault(f"{ds}/delta1_metric", []).append(res_metric["delta < 1.25"])
        # RMSE in metres: absolute error magnitude the relative AbsRel hides --
        # the calibration number for a metric-conditioned model.
        out.setdefault(f"{ds}/rmse_metric", []).append(res_metric["RMSE"])

        aligned = aligned.reshape(S, H, W).numpy()
        mask_np = mask.numpy()
        i2l = [_img2lidar(K[b, i], pose[b, i]) for i in range(S)]
        pairs = [
            tae(
                aligned[i],
                mask_np[i],
                i2l[i],
                aligned[i + 1],
                mask_np[i + 1],
                i2l[i + 1],
            )
            for i in range(S - 1)
        ]
        # tae -> (mean-abs, mean-sq) of the relative frame-to-frame error; tae_sq
        # is the spike-sensitive squared companion (finiteness of the two
        # coincides, but filter each independently to be safe).
        abs_errs = [a for a, _ in pairs if np.isfinite(a)]
        sq_errs = [s for _, s in pairs if np.isfinite(s)]
        if abs_errs:
            out.setdefault(f"{ds}/tae", []).append(float(np.mean(abs_errs)))
        if sq_errs:
            out.setdefault(f"{ds}/tae_sq", []).append(float(np.mean(sq_errs)))
    return out


def _streaming_depth_metrics(
    views: list[dict], preds: list[dict]
) -> dict[str, list[float]]:
    """Causal variant for the streaming eval: no metric sees a future frame.
    AbsRel / delta<1.25 are per-frame (metric = raw pred; affine = per-frame
    lstsq), and TAE compares consecutive RAW predictions -- only the prior
    frame is retained, mirroring deployment, and no alignment jitter leaks in.
    Returns per-clip value lists, like _val_depth_metrics."""
    pred, gt, valid, K, pose = _stack_depth_batch(views, preds)

    out: dict[str, list[float]] = {}
    B, S, H, W = gt.shape
    for b in range(B):
        ds = _clip_dataset(views, b)
        frame_stats: dict[str, list[float]] = {}
        errs: list[float] = []
        sq_errs: list[float] = []
        prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for i in range(S):
            mask = valid[b, i] & (gt[b, i] > 0)  # [H,W]
            if not mask.any():
                continue
            res_affine, _, _, _ = depth_evaluation(
                pred[b, i], gt[b, i], custom_mask=mask, scale_and_shift=True
            )
            res_metric, _, _, _ = depth_evaluation(
                pred[b, i], gt[b, i], custom_mask=mask, metric_scale=True
            )
            frame_stats.setdefault("absrel_affine", []).append(res_affine["Abs Rel"])
            frame_stats.setdefault("delta1_affine", []).append(
                res_affine["delta < 1.25"]
            )
            frame_stats.setdefault("rmse_affine", []).append(res_affine["RMSE"])
            frame_stats.setdefault("absrel_metric", []).append(res_metric["Abs Rel"])
            frame_stats.setdefault("delta1_metric", []).append(
                res_metric["delta < 1.25"]
            )
            # RMSE in metres: absolute error the relative AbsRel hides
            frame_stats.setdefault("rmse_metric", []).append(res_metric["RMSE"])

            cur = (
                pred[b, i].numpy(),
                mask.numpy(),
                _img2lidar(K[b, i], pose[b, i]),
            )
            if prev is not None:
                err, err_sq = tae(*prev, *cur)
                if np.isfinite(err):
                    errs.append(err)
                if np.isfinite(err_sq):
                    sq_errs.append(err_sq)
            prev = cur
        for k, v in frame_stats.items():
            out.setdefault(f"{ds}/{k}", []).append(float(np.mean(v)))
        if errs:
            out.setdefault(f"{ds}/tae", []).append(float(np.mean(errs)))
        if sq_errs:
            out.setdefault(f"{ds}/tae_sq", []).append(float(np.mean(sq_errs)))
    return out


# the depth metrics, in one canonical order. The per-dataset accumulator keys
# are "<dataset>/<metric>"; the blended (dataset-agnostic) logged keys are just
# "<metric>".
_DEPTH_METRIC_KEYS = (
    "absrel_affine",
    "delta1_affine",
    "rmse_affine",
    "absrel_metric",
    "delta1_metric",
    "rmse_metric",
    "tae",
    "tae_sq",
)


def _reduce_depth_metrics(
    sums: dict[str, float], counts: dict[str, int], accelerator: Accelerator
) -> dict[str, float]:
    """Cross-rank mean of the per-dataset depth-metric accumulators, plus a
    dataset-blended value per metric. Input keys are "<dataset>/<metric>".

    The reduced key set is UNIONED across ranks first: the accumulators are
    data-dependent (a rank's shard may miss a whole dataset, or a metric that
    never fired -- e.g. no finite TAE pair), so a per-rank key list would give
    mismatched collective shapes and hang NCCL. Unioning yields one fixed order
    every rank agrees on. Keys with zero global count are dropped.

    Returns log-ready keys: "<dataset>_<metric>" (per dataset) and "<metric>"
    (blended over datasets, count-weighted -- what checkpoint selection reads)."""
    keys = sorted(set(sums) | set(counts))
    if accelerator.num_processes > 1:
        from accelerate.utils import gather_object

        # [keys] per rank -> list of each rank's key list -> flatten to the union
        keys = sorted({k for part in gather_object([keys]) for k in part})
    if not keys:
        return {}

    t = torch.tensor(
        [[sums.get(k, 0.0), counts.get(k, 0)] for k in keys],
        dtype=torch.float64,
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        accelerator.reduce(t, reduction="sum")

    out: dict[str, float] = {}
    blended_sum: dict[str, float] = defaultdict(float)
    blended_cnt: dict[str, float] = defaultdict(float)
    for i, key in enumerate(keys):
        s, c = t[i, 0].item(), t[i, 1].item()
        dataset, metric = key.split("/", 1)
        blended_sum[metric] += s
        blended_cnt[metric] += c
        if c > 0:
            out[f"{dataset}_{metric}"] = s / c
    for metric, c in blended_cnt.items():
        if c > 0:
            out[metric] = blended_sum[metric] / c
    return out


def _log_val_stats(
    metric_logger: misc.MetricLogger,
    depth_avgs: dict[str, float],
    accelerator: Accelerator,
    prefix: str,
    step: int,
) -> dict:
    """Whole-epoch avg/med aggregation + wandb logging, ported from
    finetune.py::test_one_epoch. Loss meters get avg (globally reduced) and
    med (rank-local -- SmoothedValue only syncs count/total); depth metrics
    arrive pre-reduced as plain avgs."""
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)

    results = {}
    for name, meter in metric_logger.meters.items():
        if meter.count == 0:
            continue
        results[f"{name}_avg"] = meter.global_avg
        if len(meter.deque):
            results[f"{name}_med"] = meter.median
    results.update({f"{k}_avg": v for k, v in depth_avgs.items()})

    log_dict = {}
    for name, val in results.items():
        if isinstance(val, torch.Tensor) and val.ndim > 0:
            continue
        if isinstance(val, dict):
            continue
        log_dict[prefix + "/" + name] = val
    accelerator.log(misc.aggregate_per_view_metrics(log_dict), step=step)
    return results


@torch.no_grad()
def val_loop(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    args: FinetuneDepthCfg,
    mcfg: MetricCfg,
    prefix: str = "val",
) -> dict:
    """Validation on the training forward path: loss_of_one_batch with
    inference=False gives the criterion loss AND the predictions for the depth
    metrics in a single pass. Mirrors finetune.py::test_one_epoch."""
    if not torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("TF32 matmul must stay enabled (set at module import)")

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Val Epoch: [{}]".format(epoch)
    # pin to epoch 0 every pass: the sampler seeds its rng from epoch + 788
    # and ResizedDataset its slot mapping from epoch + 777, so this yields the
    # identical clip set and order on every validation
    _set_data_epoch(data_loader, 0)
    depth_sums: dict[str, float] = defaultdict(float)
    depth_counts: dict[str, int] = defaultdict(int)
    glb_exported = 0

    # fork + fix the torch RNG: the sparse-conditioning masks (torch.rand) and
    # query-point sampling are identical every epoch, and the training RNG
    # stream is restored on exit
    devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed)
        if devices:
            torch.cuda.manual_seed_all(args.seed)
        for _, batch in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
        ):
            _prepare_batch(batch, mcfg)
            result = loss_of_one_batch(
                batch,
                model,
                criterion,
                accelerator,
                inference=False,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
            )
            loss_value, loss_details = result["loss"]
            metric_logger.update(loss=float(loss_value), **loss_details)
            for k, vals in _val_depth_metrics(result["views"], result["pred"]).items():
                depth_sums[k] += float(np.sum(vals))
                depth_counts[k] += len(vals)
            # only the FINAL epoch's val pass writes GLBs -- a quick end-of-run
            # sanity check, not one set per epoch (visualize_depth.py can render
            # any checkpoint, incl. best, on demand). streaming_eval writes its
            # own single set after training.
            if args.export_glb and epoch == args.epochs - 1:
                glb_exported += _export_eval_glbs(
                    result["views"],
                    result["pred"],
                    args.output_dir,
                    tag=prefix,
                    accelerator=accelerator,
                    start_idx=glb_exported,
                    max_clips=args.export_glb_max_clips,
                )
            del result, batch

    depth_avgs = _reduce_depth_metrics(depth_sums, depth_counts, accelerator)
    results = _log_val_stats(metric_logger, depth_avgs, accelerator, prefix, step)
    model.train(True)
    return results


@torch.no_grad()
def streaming_eval(
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    step: int,
    args: FinetuneDepthCfg,
    mcfg: MetricCfg,
    prefix: str = "final_stream",
) -> dict:
    """One-shot post-training eval on the per-frame KV-cache path
    (MetricStreamVGGT.inference), with causal metrics only. The inference
    branch of loss_of_one_batch returns dict(views, pred) with no loss key,
    so no criterion runs here. Keys are namespaced by the prefix, keeping
    them apart from the non-causal val_* series.

    data_loader must yield ONE clip per batch (run() builds a dedicated
    batch-1 loader over the val config): StreamVGGT.inference folds the batch
    dim of frame["img"] into the sequence, so B>1 would silently interleave
    clips into one KV-cache stream. Checked per batch below."""
    model.eval()
    net = accelerator.unwrap_model(model)  # the DDP wrapper has no .inference
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Streaming eval:"
    _set_data_epoch(data_loader, 0)  # same epoch-0 pin as val_loop
    depth_sums: dict[str, float] = defaultdict(float)
    depth_counts: dict[str, int] = defaultdict(int)
    glb_exported = 0

    devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed)
        if devices:
            torch.cuda.manual_seed_all(args.seed)
        for _, batch in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
        ):
            if batch[0]["img"].shape[0] != 1:
                raise ValueError(
                    "streaming_eval needs a batch-1 loader (got batch size "
                    f"{batch[0]['img'].shape[0]}): StreamVGGT.inference treats "
                    "the batch dim of frame['img'] as extra frames"
                )
            _prepare_batch(batch, mcfg)
            result = loss_of_one_batch(
                batch,
                net,
                None,
                accelerator,
                inference=True,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
            )
            stats = _streaming_depth_metrics(result["views"], result["pred"])
            for k, vals in stats.items():
                depth_sums[k] += float(np.sum(vals))
                depth_counts[k] += len(vals)
            if args.export_glb:
                glb_exported += _export_eval_glbs(
                    result["views"],
                    result["pred"],
                    args.output_dir,
                    tag=prefix,
                    accelerator=accelerator,
                    start_idx=glb_exported,
                    max_clips=args.export_glb_max_clips,
                )
            del result, batch

    depth_avgs = _reduce_depth_metrics(depth_sums, depth_counts, accelerator)
    return _log_val_stats(metric_logger, depth_avgs, accelerator, prefix, step)


def main(cfg: FinetuneDepthCfg) -> None:
    mcfg = MetricCfg(
        depth_cond=cfg.depth_cond,
        lora=cfg.lora,
        encoder_cache=cfg.encoder_cache,
        train=cfg.train,
    ).validate()

    manifest = build_manifest(cfg)
    run_hash = experiment_hash(manifest)  # full, canonical identity (record + wandb)
    run_id = experiment_id(manifest)  # short display id, single source of truth
    cfg.output_dir = resolve_output_dir(cfg, run_id)
    print(f"Experiment {cfg.exp_group} id {run_id} -> {cfg.output_dir}")

    run(cfg, mcfg, manifest, run_hash, run_id)


if __name__ == "__main__":
    main(tyro.cli(FinetuneDepthCfg))
