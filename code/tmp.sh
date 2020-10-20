#!/usr/bin/env bash

INPUTDIR=/home/linlin.liu/research/ct/data/BIPED/edges/
VALID=/home/linlin.liu/research/ct/data/BIPED/edges/
OUTPUT=/home/linlin.liu/research/ct/data/model/checkpoints.biped
export CUDA_VISIBLE_DEVICES=4

python main.py \
    --input-dir $INPUTDIR \
    --input-val-dir $VALID \
    --test_list test_rgb.lst \
    --output_dir $OUTPUT \
    --test_data BIPED \
    --gpu 1

