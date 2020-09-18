#!/bin/bash
module load tensorflow
python ./run_main.py --input ./BSD300/train/ --berkeley_gradients ./BSD300/train-gt/ --sobel_path ./BSD300/train-sobel/
