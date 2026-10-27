#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=dataset_gen_llama70b
#SBATCH --output=logs/dataset_gen_%j.out
#SBATCH --error=logs/dataset_gen_%j.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00

# Merge the manual entries, then generate contexts and false answers with llama-3.1-70B.
# No resuming: an interrupted job has to be resubmitted and starts from scratch.
module load python
source .venv/bin/activate
nvidia-smi
python 02_context_generation.py