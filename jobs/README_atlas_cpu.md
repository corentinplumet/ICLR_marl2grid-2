# Atlas CPU PBS workflow

This repository trains from `main.py`. The CPU PBS launcher is
`jobs/train_cpu_atlas.pbs`.

## 1. Log in to Atlas

Use `atlas8` or `atlas9` as a login node. They are entry points for editing,
copying files, preparing environments, and submitting PBS jobs; PBS chooses the
compute node after `qsub`.

```bash
ssh <nusnet_id>@atlas8.nus.edu.sg
```

Atlas login messages from 2025 say `atlas6` and `atlas7` login nodes are
disabled, and jobs should use the `parallel` or `serial` queue. Avoid submitting
jobs from `/scratch2`; it is visible only on the `atlas9` login node and not on
compute nodes.

## 2. Copy or clone the project on Atlas

Log in to an Atlas login node, then put the repo in a working directory, usually
under `/hpctmp/$USER` because `/home` has a smaller quota.

```bash
mkdir -p /hpctmp/$USER
cd /hpctmp/$USER
git clone <your-repo-url> ICLR_marl2grid-2
cd ICLR_marl2grid-2
```

If you are not using git, copy the folder with `scp` or `rsync`.

## 3. Create the conda environment once

Do this once on Atlas, not inside every training job.

```bash
module avail conda
module load miniconda/4.12
conda env create -f conda_env.yml
conda activate marl2grid
pip install .
```

Then install the Grid2Op version required by the project README. If the package
must be built on a compute node, create a small PBS install job and run the same
`pip install ...` command from that job.

## 4. Edit the PBS account/project

Open `jobs/train_cpu_atlas.pbs` and replace:

```bash
#PBS -P CHANGE_ME_PROJECT
```

with your Atlas project/account name.

## 5. Submit a small smoke test

Start with a short run before launching the full training.

```bash
TOTAL_TIMESTEPS=2000 N_ENVS=4 qsub jobs/train_cpu_atlas.pbs
```

Check status:

```bash
qstat
qstat -f <job_id>
```

Read logs:

```bash
tail -f logs/stdout.<job_id>
tail -f logs/stderr.<job_id>
```

## 6. Submit the real training

Use the defaults:

```bash
qsub jobs/train_cpu_atlas.pbs
```

Or override parameters from the submit command:

```bash
ALG=QPLEX ENV_ID=bus36 SEED=1 N_ENVS=20 TOTAL_TIMESTEPS=25000000 qsub jobs/train_cpu_atlas.pbs
```

For extra Python flags:

```bash
EXTRA_ARGS="--difficulty 1 --use-heuristic False" qsub jobs/train_cpu_atlas.pbs
```

## Notes

- Current Atlas login messages say legacy queues such as `parallel24` are
  disabled. Use `parallel` for this CPU training job.
- The script defaults to `N_ENVS=20` and `TORCH_THREADS=1` to avoid oversubscribing
  the node.
- For this codebase, `N_ENVS` must divide algorithm frequencies. Safe values are
  `1`, `2`, `4`, `5`, `10`, and `20`.
- Checkpoints go to `checkpoint/`; logs go to `logs/`.
- The default is `--track False`, so Weights & Biases is disabled. Set
  `TRACK=True WANDB_MODE=online` only after confirming Atlas compute nodes can
  reach W&B and you have logged in with `wandb login`.
