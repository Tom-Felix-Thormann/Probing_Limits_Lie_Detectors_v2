#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=probes
#SBATCH --output=logs/05_%j.out
#SBATCH --error=logs/05_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# Merges the per-model results of step 03,  then runs step 05 (probes)
# for the four probed models (PROBE_MODELS in llm_utils.py: all but the two 7B models).
# Assumes that all tasks of run_03.sh and run_04.sh have finished successfully.
# Submit with: sbatch slurm/run_05.sh
# or right away, waiting for both array jobs: sbatch --dependency=afterok:<03 job id>:<04 job id> slurm/run_05.sh

module load python
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE  # threads for numpy / scikit-learn

python 03_forced_choice_task.py --combine
python 05_probe_training_and_testing.py