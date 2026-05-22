#!/bin/bash
# Short academic JED benchmark for finding the fastest local CPU layout.
# It runs several small no-W&B/no-checkpoint jobs on one full node and reports
# the best observed SPS from the training logs.
#SBATCH --job-name=marl2grid_sps_tune
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=7000M
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  sbatch jobs/tune_jed_sps.sbatch
  sbatch --export=ALL,ALG=MAPPO,ENV_ID=bus14 jobs/tune_jed_sps.sbatch
  sbatch --export=ALL,ALG=QPLEX,CONFIG=configs/jed/qplex_bus14_academic.toml jobs/tune_jed_sps.sbatch

Environment overrides:
  SPS_GRID          Space-separated N_ENVS:N_THREADS pairs.
                    Default: "32:4 48:4 56:4 64:4 68:2 72:1"
  PROBE_TIMESTEPS   Short run timesteps per pair. Default: 200000
  ALG               MAPPO, QPLEX, or LAGRMAPPO. Default: MAPPO
  ENV_ID            Grid2Op environment. Default: bus14
  CONFIG            TOML config to reuse. Default: configs/jed/mappo_bus14_academic.toml
  TRAIN_SCRIPT      Launcher to reuse. Default: jobs/job_jed.sh

The fastest layout is printed at the end with the matching frequency overrides.
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/job_jed.sh}"
CONFIG="${CONFIG:-configs/jed/mappo_bus14_academic.toml}"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_DIR="${PROJECT_DIR}/outputs/jed-sps-tune-${JOB_ID}"
mkdir -p "${RUN_DIR}"

SPS_GRID="${SPS_GRID:-32:4 48:4 56:4 64:4 68:2 72:1}"
PROBE_TIMESTEPS="${PROBE_TIMESTEPS:-200000}"
ALG="${ALG:-MAPPO}"
ENV_ID="${ENV_ID:-bus14}"

best_sps=0
best_pair=""
best_n_steps=""
best_eval_freq=""
best_train_freq=""
best_tg_qnet_freq=""

echo "SPS tuning run dir: ${RUN_DIR}"
echo "Grid: ${SPS_GRID}"
echo "Probe timesteps per pair: ${PROBE_TIMESTEPS}"
echo "Algorithm: ${ALG}, env: ${ENV_ID}"
echo "Config: ${CONFIG}"

for pair in ${SPS_GRID}; do
  envs="${pair%%:*}"
  threads="${pair##*:}"
  log_file="${RUN_DIR}/n_envs_${envs}_threads_${threads}.log"
  rollout_steps="$(((((20000 + envs - 1) / envs + envs - 1) / envs) * envs))"
  rollout_eval_freq="$((envs * rollout_steps))"
  train_freq="$((envs * 4))"
  qplex_eval_freq="$((train_freq * 20))"
  tg_qnet_freq="$((envs * 40))"

  echo "===== probing N_ENVS=${envs}, N_THREADS=${threads} ====="
  (
    export ALG ENV_ID
    export N_ENVS="${envs}"
    export N_THREADS="${threads}"
    export RESERVED_CPUS=0
    export TOTAL_TIMESTEPS="${PROBE_TIMESTEPS}"
    export CHECKPOINT=false
    export TRACK=false
    export WANDB_MODE=offline
    export PY_TIME_LIMIT=25
    export N_STEPS="${rollout_steps}"
    export EVAL_FREQ="${rollout_eval_freq}"
    export TRAIN_FREQ="${train_freq}"
    export TG_QNET_FREQ="${tg_qnet_freq}"
    export SEED=0
    if [ "${ALG^^}" = "QPLEX" ]; then
      export EVAL_FREQ="${qplex_eval_freq}"
    fi
    bash "${TRAIN_SCRIPT}" "${CONFIG}"
  ) > "${log_file}" 2>&1

  sps="$(grep -Eo 'SPS=[0-9]+' "${log_file}" | tail -n 1 | cut -d= -f2 || true)"
  if [ -z "${sps}" ]; then
    sps=0
  fi

  echo "N_ENVS=${envs}, N_THREADS=${threads}, SPS=${sps}, log=${log_file}"
  if [ "${sps}" -gt "${best_sps}" ]; then
    best_sps="${sps}"
    best_pair="${pair}"
    best_n_steps="${rollout_steps}"
    best_eval_freq="${rollout_eval_freq}"
    best_train_freq="${train_freq}"
    best_tg_qnet_freq="${tg_qnet_freq}"
    if [ "${ALG^^}" = "QPLEX" ]; then
      best_eval_freq="${qplex_eval_freq}"
    fi
  fi
done

if [ -n "${best_pair}" ]; then
  best_envs="${best_pair%%:*}"
  best_threads="${best_pair##*:}"
  echo "===== best observed layout ====="
  echo "N_ENVS=${best_envs}"
  echo "N_THREADS=${best_threads}"
  echo "SPS=${best_sps}"
  echo "Submit training with:"
  if [ "${ALG^^}" = "QPLEX" ]; then
    echo "sbatch --export=ALL,N_ENVS=${best_envs},N_THREADS=${best_threads},TRAIN_FREQ=${best_train_freq},EVAL_FREQ=${best_eval_freq},TG_QNET_FREQ=${best_tg_qnet_freq} jobs/job_jed.sh ${CONFIG}"
  else
    echo "sbatch --export=ALL,N_ENVS=${best_envs},N_THREADS=${best_threads},N_STEPS=${best_n_steps},EVAL_FREQ=${best_eval_freq} jobs/job_jed.sh ${CONFIG}"
  fi
else
  echo "No SPS value was observed. Check logs in ${RUN_DIR}."
fi
