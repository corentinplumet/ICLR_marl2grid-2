# JED Run Configs

These TOML files contain the parameters for a JED training run.

Submit the default MAPPO config:

```bash
sbatch jobs/job_jed.sh
```

Submit a specific config:

```bash
sbatch jobs/job_jed.sh configs/jed/qplex_bus14_academic.toml
```

Append one-off command-line overrides after the config path:

```bash
sbatch jobs/job_jed.sh configs/jed/mappo_bus14_academic.toml --difficulty 1
```

For Slurm arrays, `seed_from_slurm_array = true` makes each task use its array
index as the seed:

```bash
sbatch --array=0-7 jobs/job_jed.sh configs/jed/mappo_bus14_academic.toml
```

Environment variables can override values from `[args]` by using the uppercase
name, for example `N_THREADS`, `TOTAL_TIMESTEPS`, `ALG`, `ENV_ID`, or `SEED`.
If you override `N_ENVS`, also override the dependent frequencies:

- MAPPO/LAGRMAPPO: `N_STEPS` and `EVAL_FREQ`
- QPLEX: `TRAIN_FREQ`, `EVAL_FREQ`, and usually `TG_QNET_FREQ`


```bash
sbatch jobs/job_jed.sh configs/jed/mappo_bus14_best.toml
```