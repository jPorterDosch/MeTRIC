# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.datasets import get_data_loader
from streamvggt.loss.loss import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import pathlib
import random
import builtins
import shutil

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

from vggt.models.vggt import VGGT
from streamvggt.models.streamvggt import StreamVGGT

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

WANDB_PROJECT = "MeTRIC"
WANDB_ENTITY = "jporterdosch-university-of-tennessee-knoxville"


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def train(args):

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

    wandb_init_kwargs = {
        "name": args.exp_name,
        "dir": args.output_dir,
    }
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

    # auto resume
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # training dataset and loader
    printer.info("Building train dataset %s", args.train_dataset)
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length,
    )
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True,
        )
        for dataset in args.test_dataset.split("+")
    }

    # model
    printer.info("Loading model")
    model = StreamVGGT()
    teacher = VGGT()

    # model: PreTrainedModel = eval(args.model)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")

    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(
        f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
    )
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.long_context:
        model.fixed_input_length = False

    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        printer.info(model.load_state_dict(ckpt, strict=True))
        del ckpt  # in case it occupies memory

    printer.info("Loading teacher model")
    ckpt_teacher = torch.load(args.pretrained, map_location=device)
    teacher.load_state_dict(ckpt_teacher, strict=True)
    teacher = teacher.to("cuda")
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    del ckpt_teacher

    # freeze
    printer.info("Freezing patch embedding and positional encoding parameters...")
    frozen_params = 0
    total_params = 0

    frozen_param_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = True

    if hasattr(model, "aggregator") and hasattr(model.aggregator, "patch_embed"):
        for param in model.aggregator.patch_embed.parameters():
            if param.requires_grad:
                param.requires_grad = False

    if hasattr(model, "aggregator") and hasattr(model.aggregator, "camera_token"):
        model.aggregator.camera_token.requires_grad = False

    if hasattr(model, "aggregator") and hasattr(model.aggregator, "register_token"):
        model.aggregator.register_token.requires_grad = False

    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen_params += p.numel()
            frozen_param_names.append(name)

    printer.info(
        f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters. ({frozen_params / total_params:.2%})"
    )
    printer.info(
        f"Trainable parameters: {total_params - frozen_params:,} ({(total_params - frozen_params) / total_params:.2%})"
    )
    if frozen_param_names:
        printer.info(
            f"Example frozen parameters: {', '.join(frozen_param_names[:5])}{'...' if len(frozen_param_names) > 5 else ''}"
        )

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
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

    def write_log_stats(epoch, train_stats, test_stats):
        if accelerator.is_main_process:
            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far, data_iter_step):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):
        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far, args.start_step)

        new_best = False

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far, args.start_step)
            if new_best:
                save_model(epoch - 1, "best", best_so_far, args.start_step)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            teacher,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            args=args,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)
    accelerator.end_training()


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(
    dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False
):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length,
    )
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far, data_iter_step):
        unwrapped_model = accelerator.unwrap_model(model)
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=unwrapped_model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    start_step = args.start_step

    data_iter = metric_logger.log_every(
        data_loader, args.print_freq, accelerator, header
    )

    for data_iter_step, batch in enumerate(data_iter):
        with accelerator.accumulate(model):
            # change the range of the image to [0, 1]
            if isinstance(batch, dict) and "img" in batch:
                batch["img"] = (batch["img"] + 1.0) / 2.0
            elif isinstance(batch, list) and all(
                isinstance(v, dict) and "img" in v for v in batch
            ):
                for view in batch:
                    view["img"] = (view["img"] + 1.0) / 2.0

            epoch_f = epoch + data_iter_step / len(data_loader)
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)

            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))

            result = loss_of_one_batch(
                batch,
                model,
                criterion,
                accelerator,
                teacher=teacher,
                inference=False,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
            )

            loss, loss_details = result["loss"]  # criterion returns two values

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

            # is_metric = batch[0]["is_metric"]
            # curr_num_view = len(batch)

            del loss

            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)
            #
            metric_logger.update(loss=loss_value, **loss_details)
            #
            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                log_dict = {
                    "train_loss": loss_value_reduce,
                    "train_lr": lr,
                    "epoch": epoch_f,
                }
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_dict["train_" + name] = val
                accelerator.log(misc.aggregate_per_view_metrics(log_dict), step=step)

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch - 1, "last", float("inf"), data_iter_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)

    # Always log once at the end of the epoch, using the epoch-averaged (global_avg)
    # value for each metric rather than a single noisy last-batch value. This also
    # guarantees at least one log per epoch even if print_freq never divides evenly
    # into the number of steps (e.g. short smoke-test runs).
    epoch_log_dict = {}
    for name, meter in metric_logger.meters.items():
        if name == "step":
            continue
        elif name == "epoch":
            key = "epoch"
        else:
            key = f"train_{name}"
        epoch_log_dict[key] = meter.global_avg
    accelerator.log(misc.aggregate_per_view_metrics(epoch_log_dict), step=step)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
