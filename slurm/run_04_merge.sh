#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=merge_answers
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Combines all per-model answers into dataset/full_dataset_generated.csv.
# Needs no GPU; can also be run directly: python 04_answer_generation_task.py merge
module load python
source .venv/bin/activate
python 04_answer_generation_task.py merge
