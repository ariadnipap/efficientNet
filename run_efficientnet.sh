#!/bin/bash
SPECTROGRAM_DIR="/home/ubuntu/data/spectrograms"
JSON_DIR="/home/ubuntu/data"

python main.py \
    --spectrogram_dir $SPECTROGRAM_DIR \
    --json_dir $JSON_DIR \
    --model efficientnet_b0 \
    --optim nesterov \
    --lr 0.05 \
    --wd 1e-5 \
    --batch_size 128 \
    --epochs 30 \
    --eval_freq 5 \
    --pretrained \
    --mixup \
    --sound_aug \
    --seed 0 \
    --exp_name efficientnet_b0_run \
    --mode train