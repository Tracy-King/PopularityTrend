#!/bin/bash

#$-l rt_C.large=1
#$-l h_rt=12:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
source ~/youtuber_venv/bin/activate
module load gcc/12.2.0 python/3.10/3.10.10 cuda/12.1/12.1.1 cudnn/8.9/8.9.1

python3 preprocess.py --period w --year 2021
