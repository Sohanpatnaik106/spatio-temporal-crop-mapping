#!/bin/bash
#SBATCH --job-name=crop-mapping
#SBATCH --partition=gpu-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=2:00:00
#SBATCH --array=0

module load anaconda/3
conda activate btp

python main.py