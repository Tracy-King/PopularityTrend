#!/bin/bash

#$-l rt_=C.small
#$-l h_rt=12:00:00
#$-j y
#$-cwd

source ~/youtuber_venv/bin/activate
module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.2/8.2.4

python3 test.py --period d
