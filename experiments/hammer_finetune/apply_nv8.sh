#!/bin/bash
# Fallback if the LoRA arms OOM on L40S at num_views=10: drop every ablation
# arm to 8-frame clips (~20% less activation memory, faster steps). Applies
# --train-dataset.num-views 8 and --train.clip-len 8 to all four launch
# scripts. clip-len is a dead knob (nothing reads cfg.train.clip_len) but it
# is part of the experiment hash/manifest, so keep it in sync with num-views
# rather than letting the manifest lie.
#
# Val side needs NO change: val_dataset.num_views=4 (< 8) and there is no
# val clip_len.
#
# Idempotent; run once, then resubmit all four arms.
set -euo pipefail
cd "$(dirname "$0")"

for f in train_hammer_tokeninject_lora.sh train_hammer_tokeninject_headonly.sh \
         train_hammer_headinject_lora.sh train_hammer_headinject_headonly.sh; do
    if grep -q -- "--train-dataset.num-views" "$f"; then
        echo "$f: already applied, skipping"
        continue
    fi
    sed -i 's|^\( *\)--train-dataset.epoch-size 4500 \\|\1--train-dataset.epoch-size 4500 \\\n\1--train-dataset.num-views 8 \\\n\1--train.clip-len 8 \\|' "$f"
    grep -q -- "--train-dataset.num-views 8" "$f" || { echo "$f: FAILED to apply"; exit 1; }
    bash -n "$f"
    echo "$f: applied (num-views 8, clip-len 8)"
done
echo "done -- resubmit all four arms"
