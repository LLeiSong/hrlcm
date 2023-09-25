#!/bin/bash
#SBATCH -c 35
#SBATCH --mem 400G
#SBATCH -p general
#SBATCH -t 01-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lsong36@asu.edu

YEAR=$1

module load mamba/latest
source activate pytorch_spatial
cd ~/hrlcm

srun python hrlcm/utils/get_mean_sd_norm.py --year $YEAR