#!/usr/bin/env bash

INPUTDIR=/home/linlin.liu/research/ct/data/portrait2/train_more
VALID=/home/linlin.liu/research/ct/data/portrait2/train_more
OUTPUT=/home/linlin.liu/research/ct/data/model/checkpoints_more
export CUDA_VISIBLE_DEVICES=6

python main.py \
    --input-dir $INPUTDIR \
    --input-val-dir $VALID \
    --test_list test_rgb.lst \
    --output_dir $OUTPUT \
    --test_data BIPED \
    --gpu 1

