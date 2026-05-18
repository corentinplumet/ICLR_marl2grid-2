#!/bin/bash
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --output=routput_jobs/job_out_%j.log
#SBATCH --error=routput_jobs/job_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
if [ "$(basename "${PROJECT_DIR}")" = "jobs" ]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
fi

VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-marl2grid}"

cd "${PROJECT_DIR}"
mkdir -p routput_jobs checkpoint

if [ -d "${VENV_PATH}" ]; then
  source "${VENV_PATH}/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "No virtualenv found at ${VENV_PATH}, and conda is not available." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

CUDA="${CUDA:-false}"
CHECKPOINT="${CHECKPOINT:-true}"
N_THREADS="${N_THREADS:-1}"
N_ENVS="${N_ENVS:-40}"
N_STEPS="${N_STEPS:-520}"
EVAL_FREQ="${EVAL_FREQ:-20800}"
TIME_LIMIT="${TIME_LIMIT:-1380}"
ENV_ID="${ENV_ID:-bus14}"
ALG="${ALG:-MAPPO}"
USE_HEURISTIC="${USE_HEURISTIC:-false}"
TRACK="${TRACK:-true}"
WANDB_ENTITY="${WANDB_ENTITY:-corentin-plumet-epfl}"
WANDB_PROJECT="${WANDB_PROJECT:-marl2grid}"
WANDB_MODE="${WANDB_MODE:-online}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000000}"
GAMMA="${GAMMA:-0.99}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-80}"
N_MINIBATCHES="${N_MINIBATCHES:-4}"
ACTOR_LR="${ACTOR_LR:-3e-5}"
CRITIC_LR="${CRITIC_LR:-3e-5}"
CLIP_COEF="${CLIP_COEF:-0.2}"
SEED="${SEED:-0}"

CMD=(
  python -u main.py
  --cuda "${CUDA}"
  --checkpoint "${CHECKPOINT}"
  --n-threads "${N_THREADS}"
  --n-envs "${N_ENVS}"
  --n-steps "${N_STEPS}"
  --eval-freq "${EVAL_FREQ}"
  --time-limit "${TIME_LIMIT}"
  --env-id "${ENV_ID}"
  --alg "${ALG}"
  --use-heuristic "${USE_HEURISTIC}"
  --track "${TRACK}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --gamma "${GAMMA}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --update-epochs "${UPDATE_EPOCHS}"
  --n-minibatches "${N_MINIBATCHES}"
  --actor-lr "${ACTOR_LR}"
  --critic-lr "${CRITIC_LR}"
  --clip-coef "${CLIP_COEF}"
  --seed "${SEED}"
)

CMD+=("$@")

echo "Project dir: ${PROJECT_DIR}"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Python: $(command -v python)"
echo "SLURM job id: ${SLURM_JOB_ID:-local}"
echo "Command: ${CMD[*]}"

stdbuf -oL -eL "${CMD[@]}"

echo "Finished at: $(date)"
