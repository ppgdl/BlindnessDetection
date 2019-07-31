#!/usr/bin/env bash
BATCH_SIZE=32
GPU_ID=2,3

#python ./tools/train.py --train_path=r'./data/blindness/train_list.txt'
python train.py --gpu_id $GPU_ID --train_path '../../blindnessDectection/data/train.csv'  --batch_size $BATCH_SIZE 
