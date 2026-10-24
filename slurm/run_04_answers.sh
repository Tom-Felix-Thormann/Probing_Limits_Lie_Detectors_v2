#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=answers
#SBATCH --output=logs/answers_%x_%j.out
#SBATCH --error=logs/answers_%x_%j.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00

# Step 3 for one model: sbatch slurm/run_04_answers.sh <model_name>
# GPUs, memory and time are overridden per model in submit_all.sh.
# Resumes automatically if the job is interrupted: just submit it again.
module load python
source .venv/bin/activate
nvidia-smi
python 04_answer_generation_task.py "$1" --batch-size 32
