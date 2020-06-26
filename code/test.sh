#!/usr/bin/env bash

INPUTDIR=/home/linlin.liu/research/ct/data/BIPED/edges
OUTPUT=/home/linlin.liu/research/ct/data/model/checkpoints
CHKPNT=11/11_model.pth
RESDIR=/home/linlin.liu/research/ct/data/portrait/test_result
VALID=/home/linlin.liu/research/ct/data/portrait/test/

python test.py \
    --input-dir $INPUTDIR \
    --input-val-dir $VALID \
    --test_list test_rgb.lst \
    --output_dir $OUTPUT \
    --checkpoint_data $CHKPNT \
    --res_dir $RESDIR \
    --test_data CLASSIC \
    --gpu 1

