#!/bin/bash
# JED is a CPU-only cluster. This launcher uses one full standard node, which is
# the maximum useful allocation for the current single-node multiprocessing code.
# JED's 32-node parallel QOS would require a distributed training implementation.
#SBATCH --job-name=marl2grid_jed
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=7000M
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  sbatch jobs/train_jed.sbatch
  sbatch --export=ALL,SEED=1,ALG=QPLEX jobs/train_jed.sbatch
  sbatch jobs/train_jed.sbatch --difficulty 1 --use-heuristic False

Default allocation:
  partition=standard, qos=serial, nodes=1, cpus=72, mem-per-cpu=7000M,
  walltime=7 days.

Common environment overrides:
  PROJECT_DIR, ENV_BACKEND, CONDA_ENV_NAME, VENV_PATH, JED_MODULES,
  ALG, ENV_ID, SEED, N_ENVS, N_STEPS, EVAL_FREQ, TOTAL_TIMESTEPS,
  TRACK, WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE, EXTRA_ARGS.

Environment setup:
  ENV_BACKEND=auto   uses conda if available, otherwise .venv if present
  ENV_BACKEND=conda  activates CONDA_ENV_NAME, default marl2grid
  ENV_BACKEND=venv   activates VENV_PATH, default <project>/.venv

Notes:
  JED has no GPUs, so CUDA defaults to false.
  Keep N_STEPS and EVAL_FREQ divisible by N_ENVS for MAPPO/LAGRMAPPO.
  Keep TRAIN_FREQ and EVAL_FREQ divisible by N_ENVS for QPLEX.
EOF
  exit 0
fi

set_default() {
  local name="$1"
  local value="$2"
  if [ -z "${!name:-}" ]; then
    printf -v "${name}" '%s' "${value}"
  fi
}

add_arg() {
  local flag="$1"
  local value="${2:-}"
  if [ -n "${value}" ]; then
    ARGS+=("${flag}" "${value}")
  fi
}

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/miniforge3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
      if [ -f "${conda_sh}" ]; then
        # shellcheck source=/dev/null
        source "${conda_sh}"
        break
      fi
    done
  fi

  if ! command -v conda >/dev/null 2>&1; then
    return 1
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
}

activate_venv() {
  if [ ! -d "${VENV_PATH}" ]; then
    return 1
  fi
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
}

print_resource_summary() {
  echo "========== JED/Slurm resource summary =========="
  echo "Job id: ${SLURM_JOB_ID:-local}"
  echo "Job name: ${SLURM_JOB_NAME:-marl2grid_jed}"
  echo "Partition: ${SLURM_JOB_PARTITION:-standard}"
  echo "QOS: ${SLURM_JOB_QOS:-serial}"
  echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}"
  echo "Node list: ${SLURM_JOB_NODELIST:-$(hostname)}"
  echo "Tasks: ${SLURM_NTASKS:-1}"
  echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-72}"
  echo "CPUs on node: ${SLURM_CPUS_ON_NODE:-unknown}"
  echo "Memory per CPU: ${SLURM_MEM_PER_CPU:-7000M}"
  echo "================================================"
}

print_python_summary() {
  python - <<'PY'
import os
import sys

print("========== Python environment summary ==========")
print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
for name in ["torch", "numpy", "grid2op", "lightsim2grid", "wandb"]:
    try:
        mod = __import__(name)
        print(f"{name}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: import failed: {exc}")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Torch threads:", torch.get_num_threads())
except Exception:
    pass
print("===============================================")
PY
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
PROJECT_DIR="${PROJECT_DIR:-${SUBMIT_DIR}}"
if [ "$(basename "${PROJECT_DIR}")" = "jobs" ]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
elif [ ! -f "${PROJECT_DIR}/main.py" ]; then
  PROJECT_DIR="${SCRIPT_PROJECT_DIR}"
fi

if [ ! -f "${PROJECT_DIR}/main.py" ]; then
  echo "Could not find main.py. Set PROJECT_DIR to the repository root." >&2
  exit 1
fi

VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-marl2grid}"
ENV_BACKEND="${ENV_BACKEND:-auto}"

if command -v module >/dev/null 2>&1; then
  if [ "${PURGE_MODULES:-true}" = "true" ]; then
    module purge || true
  fi
  if [ -n "${JED_MODULES:-}" ]; then
    # shellcheck disable=SC2086
    module load ${JED_MODULES}
  fi
fi

case "${ENV_BACKEND}" in
  auto)
    if ! activate_conda; then
      activate_venv || {
        echo "Could not activate conda env '${CONDA_ENV_NAME}' or venv '${VENV_PATH}'." >&2
        echo "Set ENV_BACKEND=conda/venv, CONDA_ENV_NAME, or VENV_PATH." >&2
        exit 1
      }
    fi
    ;;
  conda)
    activate_conda || {
      echo "Could not activate conda env '${CONDA_ENV_NAME}'." >&2
      exit 1
    }
    ;;
  venv)
    activate_venv || {
      echo "Could not activate venv '${VENV_PATH}'." >&2
      exit 1
    }
    ;;
  *)
    echo "Unknown ENV_BACKEND='${ENV_BACKEND}'. Use auto, conda, or venv." >&2
    exit 1
    ;;
esac

cd "${PROJECT_DIR}"

JOB_ID="${SLURM_JOB_ID:-local}"
RUN_DIR="${PROJECT_DIR}/outputs/jed-${JOB_ID}"
mkdir -p "${RUN_DIR}" "${PROJECT_DIR}/checkpoint"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_DIR}/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
export MAX_TIME_LIMIT_MINUTES="${MAX_TIME_LIMIT_MINUTES:-10080}"
mkdir -p "${MPLCONFIGDIR}" "${WANDB_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/tmp/${SLURM_JOB_ID}" ]; then
  export TMPDIR="${TMPDIR:-/tmp/${SLURM_JOB_ID}}"
fi

SLURM_CPUS="${SLURM_CPUS_PER_TASK:-72}"
set_default CUDA false
set_default CHECKPOINT true
set_default N_THREADS 1
set_default N_ENVS "${SLURM_CPUS}"
set_default PY_TIME_LIMIT 10020
set_default ENV_ID bus14
set_default ALG MAPPO
set_default USE_HEURISTIC false
set_default TRACK true
set_default WANDB_ENTITY corentin-plumet-epfl
set_default WANDB_PROJECT marl2grid
set_default WANDB_MODE online
set_default TOTAL_TIMESTEPS 60000000
set_default GAMMA 0.99
set_default SEED 0

case "${ALG^^}" in
  MAPPO)
    set_default N_STEPS "$((N_ENVS * 4))"
    set_default EVAL_FREQ "$((N_ENVS * 280))"
    set_default MAX_GRAD_NORM 10
    set_default UPDATE_EPOCHS 80
    set_default N_MINIBATCHES 4
    set_default ACTOR_LR 3e-5
    set_default CRITIC_LR 3e-5
    set_default CLIP_COEF 0.2
    set_default INIT_DO_NOTHING_PROB 0.5
    set_default NORM_REWARD true
    ;;
  LAGRMAPPO)
    set_default N_STEPS "$((N_ENVS * 4))"
    set_default EVAL_FREQ "$((N_ENVS * 280))"
    set_default CONSTRAINTS_TYPE 1
    set_default MAX_GRAD_NORM 10
    set_default UPDATE_EPOCHS 80
    set_default N_MINIBATCHES 4
    set_default ACTOR_LR 3e-5
    set_default CRITIC_LR 3e-5
    set_default CLIP_COEF 0.2
    set_default COST_THRESHOLD 0
    set_default LAG_MUL 0.0
    set_default LAG_LR 0.05
    ;;
  QPLEX)
    set_default EVAL_FREQ "$((N_ENVS * 280))"
    set_default TRAIN_FREQ "$((N_ENVS * 4))"
    set_default TG_QNET_FREQ "$((N_ENVS * 40))"
    set_default BUFFER_SIZE 1000000
    set_default BATCH_SIZE 128
    set_default LR 3e-5
    set_default EPS_DECAY_FRAC 0.5
    ;;
esac

ARGS=(
  --cuda "${CUDA}"
  --checkpoint "${CHECKPOINT}"
  --n-threads "${N_THREADS}"
  --n-envs "${N_ENVS}"
  --eval-freq "${EVAL_FREQ}"
  --time-limit "${PY_TIME_LIMIT}"
)

add_arg --env-id "${ENV_ID:-}"
add_arg --alg "${ALG:-}"
add_arg --seed "${SEED:-}"
add_arg --constraints-type "${CONSTRAINTS_TYPE:-}"
add_arg --use-heuristic "${USE_HEURISTIC:-}"
add_arg --decentralized "${DECENTRALIZED:-}"
add_arg --track "${TRACK:-}"
add_arg --wandb-entity "${WANDB_ENTITY:-}"
add_arg --wandb-project "${WANDB_PROJECT:-}"
add_arg --wandb-mode "${WANDB_MODE:-}"
add_arg --total-timesteps "${TOTAL_TIMESTEPS:-}"
add_arg --gamma "${GAMMA:-}"
add_arg --n-steps "${N_STEPS:-}"
add_arg --max-grad-norm "${MAX_GRAD_NORM:-}"
add_arg --update-epochs "${UPDATE_EPOCHS:-}"
add_arg --n-minibatches "${N_MINIBATCHES:-}"
add_arg --actor-lr "${ACTOR_LR:-}"
add_arg --critic-lr "${CRITIC_LR:-}"
add_arg --clip-coef "${CLIP_COEF:-}"
add_arg --init-do-nothing-prob "${INIT_DO_NOTHING_PROB:-}"
add_arg --norm-reward "${NORM_REWARD:-}"
add_arg --train-freq "${TRAIN_FREQ:-}"
add_arg --tg-qnet-freq "${TG_QNET_FREQ:-}"
add_arg --buffer-size "${BUFFER_SIZE:-}"
add_arg --batch-size "${BATCH_SIZE:-}"
add_arg --lr "${LR:-}"
add_arg --eps-decay-frac "${EPS_DECAY_FRAC:-}"
add_arg --cost-threshold "${COST_THRESHOLD:-}"
add_arg --lag-mul "${LAG_MUL:-}"
add_arg --lag-lr "${LAG_LR:-}"

if [ -n "${EXTRA_ARGS:-}" ]; then
  # Use EXTRA_ARGS only for simple space-separated flags.
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi
ARGS+=("$@")

RUNNER=(python -u main.py)
if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
  RUNNER=(
    srun
    --ntasks=1
    --cpus-per-task="${SLURM_CPUS}"
    --cpu-bind=cores
    python -u main.py
  )
fi

echo "Project dir: ${PROJECT_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Started at: $(date)"
print_resource_summary
print_python_summary
echo "Algorithm: ${ALG}"
echo "Vector envs: ${N_ENVS}"
echo "Torch threads: ${N_THREADS}"
echo "Python time limit: ${PY_TIME_LIMIT} minutes"
echo "Command: ${RUNNER[*]} ${ARGS[*]}"

stdbuf -oL -eL "${RUNNER[@]}" "${ARGS[@]}"

echo "Finished at: $(date)"
