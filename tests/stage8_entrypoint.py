"""Stage 8 CHECK: the training entrypoint runs end to end.

Exercises finetune_depth.main/train() -- the path every OTHER stage test
bypasses (they call build_model + overfit_steps directly). Runs on a synthetic
in-memory dataset (no GPU dataset required) and covers the whole lifecycle:

    tyro config -> experiment hash -> resolve_output_dir -> Accelerator
    -> real DataLoader (accelerate.prepare) -> train_one_epoch
    -> mid-epoch checkpoint-last save -> epoch-boundary save -> manifest.json
    -> checkpoint-final ; then --resume -> misc.load_model -> continue -> finish

It also asserts the artifact is loadable and carries a plain argparse.Namespace
rather than the __main__-defined FinetuneDepthCfg (the checkpoint-pickle fix).

This is the "does the machine actually run" test. It does not check model
quality -- convergence on real data is a separate research result, out of scope
for a merge gate.

Run: python tests/stage8_entrypoint.py
(the orchestrator re-execs this file with --driver in a fresh subprocess so each
train() call gets a clean accelerate/singleton state; wandb is disabled.)
"""

import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")


# ---------------------------------------------------------------------------
# driver: one real run of finetune_depth.main on synthetic data (subprocess)
# ---------------------------------------------------------------------------
def driver() -> None:
    sys.path.insert(0, _SRC)
    sys.path.insert(0, _HERE)

    import finetune_depth
    from finetune_depth import FinetuneDepthCfg
    from synthetic import synthetic_loader

    # swap the real dataset construction for the synthetic loader; everything
    # downstream (accelerate.prepare, the epoch loop, saving, resume) is real.
    # build_train_loader is the module-level seam the train loop resolves at
    # call time, so replacing the attribute takes effect. It serves BOTH
    # splits, so the val/streaming passes run on synthetic data too. The
    # batch_size override (run() requests a batch-1 loader for streaming_eval
    # when args.batch_size > 1) is ignored: synthetic_loader is already
    # batch-1 by construction.
    def fake_build_train_loader(args, split, accelerator, batch_size=None):
        # 224x224 (16x16 patch grid): large enough that the track head's
        # correlation pyramid does not pool down to 0x0 (it runs because
        # loss_of_one_batch samples query points from valid_mask).
        return synthetic_loader(
            num_views=int(os.environ["SC_VIEWS"]),
            n_steps=int(os.environ["SC_STEPS"]),
            H=224,
            W=224,
        )

    finetune_depth.build_train_loader = fake_build_train_loader

    cfg = FinetuneDepthCfg(
        pretrained="",  # skip the 5GB pretrained load; this is a plumbing test
        save_dir=os.environ["SC_SAVE_DIR"],
        exp_name="stage8",
        epochs=int(os.environ["SC_EPOCHS"]),
        save_freq=0.5,  # int(0.5 * 12 steps) = 6 -> one mid-epoch save at step 6
        keep_freq=999,  # skip numbered per-epoch checkpoints (keep the test lean)
        print_freq=5,
        batch_size=1,
        num_workers=0,
        lr=1e-4,
        resume=os.environ.get("SC_RESUME") or None,
    )
    finetune_depth.main(cfg)

    if cfg.resume:
        # main() -> train() -> misc.load_model mutates cfg.start_epoch in place
        print(f"DRIVER_RESUMED start_epoch={cfg.start_epoch}")
        if cfg.start_epoch <= 0:
            raise SystemExit("resume did not advance start_epoch")


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def _run(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--driver"],
        cwd=_SRC,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )


def run_checks() -> None:
    import torch  # noqa: F401 (import here so --driver path stays light)

    with tempfile.TemporaryDirectory(prefix="stage8_", dir="/tmp") as save_dir:
        base_env = {
            **os.environ,
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "SC_SAVE_DIR": save_dir,
            "SC_VIEWS": "3",
            "SC_STEPS": "12",
        }

        # --- Run A: fresh end-to-end run (1 epoch) ---
        rA = _run({**base_env, "SC_EPOCHS": "1"})
        assert rA.returncode == 0, (
            f"run A failed:\n{rA.stdout[-4000:]}\n{rA.stderr[-4000:]}"
        )
        assert "saving at step" in rA.stdout, (
            "mid-epoch checkpoint-last save never fired"
        )
        assert "Val Epoch: [0]" in rA.stdout, "val_loop never ran (val_freq=1 default)"
        assert "Streaming eval:" in rA.stdout, "post-training streaming_eval never ran"

        dirs = [d for d in os.listdir(save_dir) if d.startswith("stage8_")]
        assert len(dirs) == 1, f"expected one output dir, got {dirs}"
        out = os.path.join(save_dir, dirs[0])
        for fname in (
            "manifest.json",
            "checkpoint-last.pth",
            "checkpoint-best.pth",
            "checkpoint-final.pth",
        ):
            assert os.path.isfile(os.path.join(out, fname)), f"missing {fname} in {out}"
        print(
            "[stage8] run A: e2e train() completed; manifest + mid-epoch + best + final checkpoints written; val + streaming eval ran"
        )

        # --- artifact is loadable from a process WITHOUT streamvggt on the path
        # (the checkpoint-pickle fix: args must reduce to builtins only) ---
        probe = (
            "import torch, argparse, sys\n"
            f"ck = torch.load({os.path.join(out, 'checkpoint-final.pth')!r}, map_location='cpu', weights_only=False)\n"
            "assert isinstance(ck['args'], argparse.Namespace), type(ck['args'])\n"
            "assert isinstance(ck['args'].depth_cond['injection'], str), ck['args'].depth_cond\n"
            "assert 'model' in ck and any('aggregator' in k for k in ck['model'])\n"
            "assert 'streamvggt' not in sys.modules, 'unpickling pulled in streamvggt'\n"
            "print('probe OK')\n"
        )
        rP = subprocess.run(
            [sys.executable, "-c", probe],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert rP.returncode == 0 and "probe OK" in rP.stdout, (
            f"checkpoint not self-contained:\n{rP.stdout[-2000:]}\n{rP.stderr[-2000:]}"
        )
        print(
            "[stage8] run A: checkpoint unpickles with builtins only, no streamvggt import (pickle fix holds)"
        )

        # --- Run B: resume from checkpoint-last, train one more epoch ---
        # resume derives the output dir from the checkpoint's parent, so Run B
        # continues INTO Run A's dir even though --epochs (an identity knob)
        # changed; capture the final-checkpoint mtime so the assertion actually
        # verifies Run B re-wrote it rather than matching Run A's leftover file.
        final_ckpt = os.path.join(out, "checkpoint-final.pth")
        mtime_a = os.path.getmtime(final_ckpt)
        rB = _run(
            {
                **base_env,
                "SC_EPOCHS": "2",
                "SC_RESUME": os.path.join(out, "checkpoint-last.pth"),
            }
        )
        assert rB.returncode == 0, (
            f"run B (resume) failed:\n{rB.stdout[-4000:]}\n{rB.stderr[-4000:]}"
        )
        assert "DRIVER_RESUMED" in rB.stdout, "resume path did not advance start_epoch"
        new_dirs = [d for d in os.listdir(save_dir) if d.startswith("stage8_")]
        assert new_dirs == dirs, (
            f"resume forked a new output dir: {set(new_dirs) - set(dirs)}"
        )
        assert os.path.getmtime(final_ckpt) > mtime_a, (
            "resume did not re-write checkpoint-final in the run's dir"
        )
        print(
            "[stage8] run B: --resume continued INTO the same dir and re-wrote checkpoint-final"
        )

    print("STAGE 8 PASS")


if __name__ == "__main__":
    if "--driver" in sys.argv:
        driver()
    else:
        run_checks()
