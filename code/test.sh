#!/usr/bin/env bash

INPUTDIR=/home/linlin.liu/research/ct/data/portrait2/train
OUTPUT=/home/linlin.liu/research/ct/data/model/checkpoints2_exp2
CHKPNT=24/24_model.pth
RESDIR=/home/linlin.liu/research/ct/data/portrait2/test_result
VALID=/home/linlin.liu/research/ct/data/portrait2/test
export CUDA_VISIBLE_DEVICES=2

python test.py \
    --input-dir $INPUTDIR \
    --input-val-dir $VALID \
    --test_list test_rgb.lst \
    --output_dir $OUTPUT \
    --checkpoint_data $CHKPNT \
    --res_dir $RESDIR \
    --test_data CLASSIC \
    --gpu 1

