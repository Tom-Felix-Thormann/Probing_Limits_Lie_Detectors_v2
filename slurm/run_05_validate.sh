#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=validate_llama70b
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --time=02:30:00

# Validates the misleading answers (generated rows + manual baseline) with llama-3.1-70B and makes the plots in figures/.
# Resumes automatically if the job is interrupted: just submit it again.
# Submit from the project root: sbatch slurm/run_05_validate.sh
module load python
source .venv/bin/activate
nvidia-smi
python 05_validate_generations.py --batch-size 16
