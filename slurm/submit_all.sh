#!/bin/bash
# Submits the whole pipeline after step 01 with its dependencies. Run it with bash (not sbatch) from the
# project directory, on the login node:
#     bash slurm/submit_all.sh                    # all steps
#     bash slurm/submit_all.sh --no-forced-choice # without step 03 (its results do not depend on step 02;
#                                                 # run_05.sh then merges the existing per-model results)
#
#     run_02.sh   dataset generation (llama-3.1-70B)          starts right away
#     run_03.sh   forced choice task, 6 models (array)        starts right away (uses only the manual rows)
#     run_04.sh   activation extraction, 4 models (array)     after run_02.sh succeeded
#     run_05.sh   merge 03, probes (CPU)         after all tasks of run_03.sh and run_04.sh succeeded
#
# If a job fails, the jobs that depend on it are cancelled (instead of pending forever). Resubmit from the
# failed step on, e.g. a single array task with: sbatch --array=<index> slurm/run_04.sh
set -euo pipefail

SLURM_DIR=$(dirname "$0")
if [[ ! -f llm_utils.py ]]; then
    echo "Run this script from the project directory: bash slurm/submit_all.sh" >&2
    exit 1
fi
mkdir -p logs

# submit SCRIPT [DEPENDENCY]  -> prints the job id
submit() {
    local options=(--parsable)
    if [[ -n "${2:-}" ]]; then
        options+=(--dependency="$2" --kill-on-invalid-dep=yes)
    fi
    local id
    id=$(sbatch "${options[@]}" "$SLURM_DIR/$1") || exit 1
    echo "${id%%;*}"  # --parsable prints "id" or "id;cluster"
}

generation=$(submit run_02.sh)
echo "run_02.sh (dataset generation):   job $generation"

activations=$(submit run_04.sh "afterok:$generation")
echo "run_04.sh (activations):          job $activations (after $generation)"

probe_dependency="afterok:$activations"
if [[ "${1:-}" != "--no-forced-choice" ]]; then
    forced_choice=$(submit run_03.sh)
    echo "run_03.sh (forced choice):        job $forced_choice"
    probe_dependency="$probe_dependency:$forced_choice"
fi

probes=$(submit run_05.sh "$probe_dependency")
echo "run_05.sh (merge, probes): job $probes (${probe_dependency#afterok:})"

echo "Check the queue with: squeue --me"