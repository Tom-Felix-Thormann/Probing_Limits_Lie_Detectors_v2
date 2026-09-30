#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --output=llm_test_%j.out
#SBATCH --error=llm_test_%j.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

module load python
source .venv/bin/activate
nvidia-smi
python 02_test_llm.py