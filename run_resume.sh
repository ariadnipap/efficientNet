#!/bin/bash
SPECTROGRAM_DIR="/home/ubuntu/data/spectrograms"
JSON_DIR="/home/ubuntu/data"
CHECKPOINT="./logs/efficientnet_b0_run/mixup/pretrained/sound_aug/efficientnet_b0/nesterov_b128_lr0.05_wd1e-05/train/2/checkpoints/checkpoint_29.pt"

python resume_training.py \
    --spectrogram_dir $SPECTROGRAM_DIR \
    --json_dir $JSON_DIR \
    --model efficientnet_b0 \
    --model_weight $CHECKPOINT \
    --optim nesterov \
    --lr 0.005 \
    --wd 1e-5 \
    --batch_size 128 \
    --epochs 20 \
    --eval_freq 5 \
    --mixup \
    --sound_aug \
    --seed 0 \
    --exp_name efficientnet_b0_resume \
    --mode train