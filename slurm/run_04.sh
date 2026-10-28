#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=forced_choice
#SBATCH --output=logs/04_%A_%a.out
#SBATCH --error=logs/04_%A_%a.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-5

# Step 04 (forced choice task) for one model per array task, all six models in parallel.
# Every task gets the resources of the largest models (4 GPUs).
# Submit with: mkdir -p logs && sbatch run_04.sh
# A single model can be resubmitted by its index, e.g.: sbatch --array=4 run_04.sh
# The per-model results are merged by run_06.sh (python 04_forced_choice_task.py --combine).
# No resuming: an interrupted task has to be resubmitted and starts from scratch.

MODELS=(
    gemma-3-27B-instruct      # 0
    llama-3.1-70B-instruct    # 1
    llama-3.1-8B-instruct     # 2
    mistral-7B-instruct       # 3
    qwen-2.5-72B-instruct     # 4
    qwen-2.5-7B-instruct      # 5
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Model: $MODEL"

module load python
source .venv/bin/activate
nvidia-smi

python 04_forced_choice_task.py --model "$MODEL"