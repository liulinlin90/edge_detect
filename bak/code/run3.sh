#!/usr/bin/env bash

INPUTDIR=/home/linlin.liu/research/ct/data/portrait2/train
VALID=/home/linlin.liu/research/ct/data/portrait2/train
OUTPUT=/home/linlin.liu/research/ct/data/model/checkpoints3
CHKPNT=finetune/24_model.pth

python main2.py \
    --input-dir $INPUTDIR \
    --input-val-dir $VALID \
    --test_list test_rgb.lst \
    --output_dir $OUTPUT \
    --test_data BIPED \
    --checkpoint_data $CHKPNT \
    --lr 1e-6 \
    --gpu 0

