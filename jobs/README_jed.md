# JED Slurm Launcher

`jobs/job_jed.sh` launches this repository's CPU training on EPFL SCITAS JED for
academic users. It reads the run parameters from TOML files in `configs/jed/`.

## Resource choice

The launcher requests one full academic JED node:

```bash
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=7000M
#SBATCH --time=12:00:00
```

This is the largest useful allocation for one training process in the current
code path. The training uses local Python multiprocessing through
`AsyncMultiAgentVecEnv`; it does not use MPI, distributed PyTorch, Ray, or
another multi-node executor. SCITAS documents the academic QOS as allowing up to
8 nodes per job, but one training process would not use those extra nodes
without a distributed implementation.

For maximum aggregate throughput across multiple seeds or configurations, use a
Slurm array of one-node jobs:

```bash
sbatch --array=0-7 jobs/job_jed.sh configs/jed/mappo_bus14_academic.toml
```

When submitted as an array, the script uses `SLURM_ARRAY_TASK_ID` as the default
seed.

## SPS tuning

The fastest `N_ENVS` is workload-dependent. The default launcher reserves a few
CPUs for the learner/logger and uses the rest as environment workers:

```bash
N_ENVS=64
N_THREADS=4
TRACK=false
WANDB_MODE=offline
```

Run a short sweep on one full node to measure SPS:

```bash
sbatch jobs/tune_jed_sps.sbatch
```

Then submit the real run with the best pair printed by the tuner:

```bash
sbatch --export=ALL,N_ENVS=<best>,N_THREADS=<best>,N_STEPS=<best>,EVAL_FREQ=<best> jobs/job_jed.sh configs/jed/mappo_bus14_academic.toml
```

You can try a custom sweep:

```bash
sbatch --export=ALL,SPS_GRID="48:4 56:4 64:4 68:2 72:1" jobs/tune_jed_sps.sbatch
```

## First-time setup on JED

```bash
ssh <username>@jed.hpc.epfl.ch
cd /scratch/$USER
git clone <your-repo-url> ICLR_marl2grid-2
cd ICLR_marl2grid-2

conda env create -f conda_env.yml
conda activate marl2grid
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

If your conda command is provided by a module, load that module before creating
the environment and pass it at submission time with `JED_MODULES`.

## Submit

```bash
sbatch jobs/job_jed.sh
```

Submit a different config:

```bash
sbatch jobs/job_jed.sh configs/jed/qplex_bus14_academic.toml
```

Override config values with exported variables or extra Python flags:

```bash
sbatch --export=ALL,SEED=1 jobs/job_jed.sh configs/jed/mappo_bus14_academic.toml --difficulty 1
```

For MAPPO and LAGRMAPPO, keep `N_STEPS` and `EVAL_FREQ` divisible by `N_ENVS`.
For QPLEX, keep `TRAIN_FREQ` and `EVAL_FREQ` divisible by `N_ENVS`.
