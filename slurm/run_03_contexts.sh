#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=contexts_llama70b
#SBATCH --output=logs/contexts_%j.out
#SBATCH --error=logs/contexts_%j.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00

# Steps 1 + 2: merge the manual entries, assign rows to models, generate contexts with llama-3.1-70B.
# Resumes automatically if the job is interrupted: just submit it again.
module load python
source .venv/bin/activate
nvidia-smi
python 03_dataset_generation_task.py --batch-size 16
