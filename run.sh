#!/usr/bin/env bash
#$ -l ram_free=10G,mem_free=10G,gpu=0
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -N qlstmTest
#$ -M sli136@jhu.edu

source /home/sli136/espnet/tools/anaconda/bin/activate espnet

echo "line 9"
python3 /home/$USER/espnet/qlstm/main.py
echo "line 11"