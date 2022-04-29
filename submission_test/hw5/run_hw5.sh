#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/21-22/575k/env
python run.py
python run.py --num_prev_chars 5
