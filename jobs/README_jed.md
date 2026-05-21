# JED Slurm Launcher

`jobs/train_jed.sbatch` launches this repository's CPU training on EPFL SCITAS
JED.

## Resource choice

The launcher requests one full standard JED node:

```bash
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=7000M
#SBATCH --time=7-00:00:00
```

This is the largest useful allocation for the current code path. The training
uses local Python multiprocessing through `AsyncMultiAgentVecEnv`; it does not
use MPI, distributed PyTorch, Ray, or another multi-node executor. JED's
`parallel` QOS can allocate more nodes, but one training process would not use
those extra nodes without a distributed implementation.

For bachelor/master academic accounts, SCITAS documents the `academic`
partition/QOS. If that is your account type, change the two resource lines to:

```bash
#SBATCH --partition=academic
#SBATCH --qos=academic
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
sbatch jobs/train_jed.sbatch
```

Override defaults with exported variables:

```bash
sbatch --export=ALL,SEED=1,ALG=QPLEX,ENV_ID=bus36 jobs/train_jed.sbatch
```

Pass extra Python flags after the script:

```bash
sbatch jobs/train_jed.sbatch --difficulty 1 --use-heuristic False
```

The default `N_ENVS` is `72`, matching the full-node CPU allocation. For MAPPO
and LAGRMAPPO, keep `N_STEPS` and `EVAL_FREQ` divisible by `N_ENVS`. For QPLEX,
keep `TRAIN_FREQ` and `EVAL_FREQ` divisible by `N_ENVS`.
