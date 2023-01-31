#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py --mlflow=143.248.148.81:5000 --conf=conf/vit.yaml