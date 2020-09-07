#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python model/train.py 2>&1 |tee ./model.log
