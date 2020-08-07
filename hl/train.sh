#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python model/train.py 2>&1 |tee ./model.log
