#!/bin/bash
#SBATCH --job-name=marl2grid_8nodes
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=7000M
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/jed/mappo_bus14_academic_8nodes.toml}"
shift || true

exec bash "${SCRIPT_DIR}/job_jed.sh" "${CONFIG}" "$@"
