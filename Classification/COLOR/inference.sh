#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python inference_for_preprocess.py --mlflow=143.248.148.81:5000 --conf=conf/inference.yaml