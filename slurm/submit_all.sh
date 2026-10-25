#!/bin/bash
# Submits the whole pipeline from the project root: bash slurm/submit_all.sh
#   1. contexts with llama-3.1-70B (4 GPUs)
#   2. one answer job per model, sized to the model, all starting once the contexts are done
#   3. a merge job once all answer jobs succeeded
# Pass --skip-contexts if generation_task.csv already has all contexts.
set -e
mkdir -p logs

dependency=""
if [ "$1" != "--skip-contexts" ]; then
    contexts_job=$(sbatch --parsable slurm/run_03_contexts.sh)
    echo "contexts: $contexts_job"
    dependency="--dependency=afterok:$contexts_job"
fi

answer_jobs=()
for model in gemma-3-27B-instruct llama-3.1-70B-instruct llama-3.1-8B-instruct mistral-7B-instruct qwen-2.5-72B-instruct qwen-2.5-7B-instruct; do
    case "$model" in
        *70B*|*72B*) resources="--gpus=4 --mem=200G --time=03:00:00" ;;
        *27B*)       resources="--gpus=2 --mem=120G --time=02:00:00" ;;
        *)           resources="--gpus=1 --mem=64G --time=01:30:00" ;;
    esac
    job=$(sbatch --parsable $dependency $resources --job-name="ans_$model" slurm/run_04_answers.sh "$model")
    echo "$model: $job"
    answer_jobs+=("$job")
done

merge_job=$(sbatch --parsable --dependency=afterok:$(IFS=:; echo "${answer_jobs[*]}") slurm/run_04_merge.sh)
echo "merge: $merge_job"
