#!/bin/bash
#SBATCH --job-name=default
#SBATCH --gres=gpu:2080:2
#SBATCH --output=/mnt/hdd0/lijunyi/slurm_logs/%j.out
#SBATCH --error=/mnt/hdd0/lijunyi/slurm_logs/%j.err

source activate
conda activate ~/anaconda/pytorch2
cd /mnt/hdd0/lijunyi/codes/TBSN
sh train.sh