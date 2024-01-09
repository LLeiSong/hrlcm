#!/bin/bash
#SBATCH -c 35
#SBATCH --mem 400G
#SBATCH -p general
#SBATCH -t 01-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lsong36@asu.edu

YEAR=$1

module load mamba/latest
# Finally get the right way to use other nodes
eval "$(conda shell.bash hook)"
source activate pytorch_spatial
cd ~/hrlcm

srun python hrlcm/utils/get_mean_sd_norm.py --year $YEAR