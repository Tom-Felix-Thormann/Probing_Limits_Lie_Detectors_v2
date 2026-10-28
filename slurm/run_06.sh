#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=probes
#SBATCH --output=logs/06_%j.out
#SBATCH --error=logs/06_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# Merges the per-model results of step 04, then runs step 06 (probes) for the four probed models
# (PROBE_MODELS in llm_utils.py: all but the two 7B models).
# Assumes that all tasks of run_04.sh and run_05.sh have finished successfully.
# Submit with: sbatch run_06.sh
# or right away, waiting for both array jobs: sbatch --dependency=afterok:<04 job id>:<05 job id> run_06.sh
# Step 06 runs on the CPU (scikit-learn); the GPU is only requested for the partition.

module load python
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE  # threads for numpy / scikit-learn

python 04_forced_choice_task.py --combine
python 06_probe_training_and_testing.py