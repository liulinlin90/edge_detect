#!/usr/bin/env bash

INPUTDIR=/home/linlin.liu/research/ct/data/portrait2/train_edge
VALID=/home/linlin.liu/research/ct/data/portrait2/train_edge
OUTPUT=/home/linlin.liu/research/ct/data/model/checkpoints2_edge

python main2.py \
    --input-dir $INPUTDIR \
    --input-val-dir $VALID \
    --test_list test_rgb.lst \
    --output_dir $OUTPUT \
    --test_data BIPED \
    --gpu 0

