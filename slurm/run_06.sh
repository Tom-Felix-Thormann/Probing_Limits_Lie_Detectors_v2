#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=probes
#SBATCH --output=logs/06_%j.out
#SBATCH --error=logs/06_%j.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# Merges the per-model results of step 04, then runs step 06 (probes) for all models.
# Assumes that all tasks of run_04_05.sh have finished successfully.
# Submit with: sbatch run_06.sh
# or right away, waiting for the array job: sbatch --dependency=afterok:<array job id> run_06.sh
# Step 06 runs on the CPU (scikit-learn); the GPU is only requested for the partition.

module load python
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE  # threads for numpy / scikit-learn

python 04_forced_choice_task.py --combine
python 06_probe_training_and_testing.py
