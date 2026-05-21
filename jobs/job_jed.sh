#!/bin/bash
#SBATCH --job-name=marl2grid_jed
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=academic
#SBATCH --qos=academic
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
  sbatch jobs/job_jed.sh
  sbatch jobs/job_jed.sh configs/jed/qplex_bus14_academic.toml
  sbatch --array=0-7 jobs/job_jed.sh configs/jed/mappo_bus14_academic.toml
  sbatch jobs/job_jed.sh configs/jed/mappo_bus14_academic.toml --difficulty 1

Environment:
  JED_CONFIG       Default config path.
  CONDA_ENV_NAME   Conda env to activate. Default: marl2grid.
  VENV_PATH        Fallback virtualenv path. Default: <repo>/.venv.
  JED_MODULES      Optional modules to load before activation.
  DRY_RUN=true     Print the resolved command without launching training.
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

CONFIG="${JED_CONFIG:-configs/jed/mappo_bus14_academic.toml}"
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
  CONFIG="$1"
  shift
fi

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [ -n "${JED_MODULES:-}" ]; then
    # shellcheck disable=SC2086
    module load ${JED_MODULES}
  fi
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-marl2grid}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"

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

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
elif [ -d "${VENV_PATH}" ]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
else
  echo "Could not find conda or virtualenv at ${VENV_PATH}." >&2
  exit 1
fi

RUN_CONFIG_ARGS=()
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  RUN_CONFIG_ARGS+=(--dry-run)
fi

python -u jobs/run_from_config.py "${RUN_CONFIG_ARGS[@]}" "${CONFIG}" "$@"
