#!/bin/bash

set -e

workdir='..'
model_name='streamvggt'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
datasets=('sintel' 'bonn' 'kitti')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    echo "$output_dir"
    pose_eval_stride=1
    if [ "$data" == "bonn" ]; then
        # full 110-frame bonn clips OOM on a 24GB GPU at stride 1; subsample frames as a workaround.
        # results for bonn are not directly comparable to a stride-1 paper baseline.
        pose_eval_stride=2
    fi
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1  ../src/eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --pose_eval_stride "$pose_eval_stride" \
        --size 518
    python ../src/eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --pose_eval_stride "$pose_eval_stride" \
    --align "scale"
done
