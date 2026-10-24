#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --output=llm_test_%j.out
#SBATCH --error=llm_test_%j.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00

module load python
source .venv/bin/activate
nvidia-smi
python 02_test_llm.py 70