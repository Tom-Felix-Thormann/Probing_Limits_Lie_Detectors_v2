#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=forced_choice_and_activations
#SBATCH --output=logs/04_05_%A_%a.out
#SBATCH --error=logs/04_05_%A_%a.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-5

# Steps 04 (forced choice task) and 05 (activation extraction) for one model per array task,
# all six models in parallel. Every task gets the resources of the largest models (4 GPUs).
# Submit with: mkdir -p logs && sbatch run_04_05.sh
# A single model can be resubmitted by its index, e.g.: sbatch --array=4 run_04_05.sh
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

# 05 also runs if 04 fails; the task fails if either step fails
status=0
python 04_forced_choice_task.py --model "$MODEL" || status=1
python 05_activation_extraction.py --model "$MODEL" || status=1
exit $status
