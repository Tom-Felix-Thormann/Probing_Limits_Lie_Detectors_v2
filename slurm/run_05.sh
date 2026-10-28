#!/bin/bash
#SBATCH --account=strascl8_0000
#SBATCH --job-name=activations
#SBATCH --output=logs/05_%A_%a.out
#SBATCH --error=logs/05_%A_%a.err
#SBATCH -N 1
#SBATCH --partition=fat_gpu_filler
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-3

# Step 05 (activation extraction) for one model per array task, the four probed models in parallel
# (PROBE_MODELS in llm_utils.py: all but the two 7B models). Every task gets the resources of the
# largest models (4 GPUs).
# Submit with: mkdir -p logs && sbatch run_05.sh
# A single model can be resubmitted by its index, e.g.: sbatch --array=3 run_05.sh
# No resuming: an interrupted task has to be resubmitted and starts from scratch.

MODELS=(
    gemma-3-27B-instruct      # 0
    llama-3.1-70B-instruct    # 1
    llama-3.1-8B-instruct     # 2
    qwen-2.5-72B-instruct     # 3
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Model: $MODEL"

module load python
source .venv/bin/activate
nvidia-smi

python 05_activation_extraction.py --model "$MODEL"