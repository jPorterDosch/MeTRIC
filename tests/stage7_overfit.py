"""Stage 7 CHECK: training wiring.

Launches src/finetune_depth.py in smoke mode twice -- injection=head and
injection=token, changing ONLY that flag -- and verifies:
  * both runs' 5-step overfit losses decrease (monotonically-ish, checked
    in-process by the entrypoint, re-checked here);
  * grad checkpointing is on;
  * the two manifests are comparable (confound rule) and the token run passes
    the startup confound check against the head run's manifest;
  * a run with a different lora.rank FAILS the confound check.
"""

import json
import os
import subprocess
import sys
import tempfile

from common import CKPT, ROOT

PY = sys.executable
SRC = os.path.join(ROOT, "src")


def launch(save_dir, injection, exp_name=None, extra=()):
    cmd = [
        PY,
        "finetune_depth.py",
        "smoke_test=true",
        f"depth_cond.injection={injection}",
        f"exp_name={exp_name or 'smoke_' + injection}",
        f"save_dir={save_dir}",
        f"pretrained={CKPT}",
        *extra,
    ]
    return subprocess.run(cmd, cwd=SRC, capture_output=True, text=True, timeout=560)


def main():
    with tempfile.TemporaryDirectory(prefix="stage7_") as tmp:
        # head arm (control)
        r = launch(tmp, "head")
        assert r.returncode == 0, (
            f"head smoke failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        )
        head_manifest = os.path.join(tmp, "smoke_head", "manifest.json")

        # token arm (proposed), only the flag changed, confound-checked at startup
        r = launch(tmp, "token", extra=(f"compare_with_manifest={head_manifest}",))
        assert r.returncode == 0, (
            f"token smoke failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        )

        for injection in ("head", "token"):
            with open(
                os.path.join(tmp, f"smoke_{injection}", "smoke_result.json")
            ) as f:
                res = json.load(f)
            losses = res["losses"]
            assert res["grad_checkpoint"] is True, "grad checkpointing must be on"
            assert losses[-1] < losses[0], (
                f"{injection}: loss did not decrease: {losses}"
            )
            rises = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
            assert rises <= 1, f"{injection}: loss not monotonically-ish: {losses}"
            print(
                f"[stage7] {injection}: {losses[0]:.4f} -> {losses[-1]:.4f} (rises {rises}) OK"
            )

        # negative test: differing lora.rank must trip the startup confound check
        r = launch(
            tmp,
            "token",
            exp_name="smoke_bad",
            extra=(
                "lora.rank=8",
                f"compare_with_manifest={head_manifest}",
            ),
        )
        assert r.returncode != 0 and "CONFOUND RULE VIOLATION" in (
            r.stdout + r.stderr
        ), "confound check did not fire for differing lora.rank"
        print(
            "[stage7] confound check fires at the entrypoint for a non-comparable run"
        )

    print("STAGE 7 PASS")


if __name__ == "__main__":
    main()
