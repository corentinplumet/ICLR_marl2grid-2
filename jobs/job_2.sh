#!/bin/bash
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --output=routput_jobs/job_downloaded_out_%j.log
#SBATCH --error=routput_jobs/job_downloaded_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
elif [ -d "${VENV_PATH}" ]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "No virtualenv found at ${VENV_PATH}, and conda is not available." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

# Defaults mirror /private/tmp/RL_Marl2grid/Topology_Task.
CUDA="${CUDA:-true}"
CHECKPOINT="${CHECKPOINT:-true}"
N_THREADS="${N_THREADS:-4}"
N_ENVS="${N_ENVS:-20}"
N_STEPS="${N_STEPS:-2000}"
EVAL_FREQ="${EVAL_FREQ:-80000}"
TIME_LIMIT="${TIME_LIMIT:-1300}"
ENV_ID="${ENV_ID:-bus14}"
ALG="${ALG:-MAPPO}"
USE_HEURISTIC="${USE_HEURISTIC:-false}"
TRACK="${TRACK:-true}"
WANDB_ENTITY="${WANDB_ENTITY:-corentin-plumet-epfl}"
WANDB_PROJECT="${WANDB_PROJECT:-marl2grid}"
WANDB_MODE="${WANDB_MODE:-online}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-25000000}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
N_MINIBATCHES="${N_MINIBATCHES:-8}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TARGET_KL="${TARGET_KL:-0.02}"
NORM_ADV="${NORM_ADV:-true}"
CLIP_COEF="${CLIP_COEF:-0.2}"
CLIP_VFLOSS="${CLIP_VFLOSS:-true}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
VF_COEF="${VF_COEF:-0.5}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
# ACTOR_LAYERS="${ACTOR_LAYERS:-256 256 256}"
# CRITIC_LAYERS="${CRITIC_LAYERS:-256 256 256}"
#A CTOR_ACT_FN="${ACTOR_ACT_FN:-relu}"
# CRITIC_ACT_FN="${CRITIC_ACT_FN:-relu}"
INIT_DO_NOTHING_PROB="${INIT_DO_NOTHING_PROB:-0.5}"
NORM_REWARD="${NORM_REWARD:-true}"
SEED="${SEED:-0}"

# read -r -a ACTOR_LAYER_ARGS <<< "${ACTOR_LAYERS}"
# read -r -a CRITIC_LAYER_ARGS <<< "${CRITIC_LAYERS}"

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
  --gae-lambda "${GAE_LAMBDA}"
  --update-epochs "${UPDATE_EPOCHS}"
  --n-minibatches "${N_MINIBATCHES}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --target-kl "${TARGET_KL}"
  --norm-adv "${NORM_ADV}"
  --clip-coef "${CLIP_COEF}"
  --clip-vfloss "${CLIP_VFLOSS}"
  --entropy-coef "${ENTROPY_COEF}"
  --vf-coef "${VF_COEF}"
  --actor-lr "${ACTOR_LR}"
  --critic-lr "${CRITIC_LR}"
  # --actor-layers "${ACTOR_LAYER_ARGS[@]}"
  # --critic-layers "${CRITIC_LAYER_ARGS[@]}"
  # --actor-act-fn "${ACTOR_ACT_FN}"
  # --critic-act-fn "${CRITIC_ACT_FN}"
  --init-do-nothing-prob "${INIT_DO_NOTHING_PROB}"
  --norm-reward "${NORM_REWARD}"
  --norm-obs "${NORM_OBS}"
  --seed "${SEED}"
)

CMD+=("$@")

echo "Project dir: ${PROJECT_DIR}"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Python: $(command -v python)"
echo "SLURM job id: ${SLURM_JOB_ID:-local}"
echo "Downloaded-repo hyperparameter launcher"
echo "Command: ${CMD[*]}"

stdbuf -oL -eL "${CMD[@]}"

echo "Finished at: $(date)"