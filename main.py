import os
from time import time

from alg.qplex.core import QPLEX
from alg.lagr_mappo.core import LagrMAPPO
from alg.mappo.core import MAPPO
from common.checkpoint import MAPPOCheckpoint, QPLEXCheckpoint, LagrMAPPOCheckpoint
from common.distributed import cleanup_distributed, init_distributed
from common.imports import *
from common.utils import set_random_seed, set_torch, str2bool
from env.config import get_env_args
from env.utils import MAEnvWrapper
from env.wrappers import AsyncMultiAgentVecEnv

# Dictionary mapping algorithm names to their corresponding classes
ALGORITHMS: Dict[str, Type[Any]] = {'QPLEX': QPLEX, 'MAPPO': MAPPO, 'LAGRMAPPO': LagrMAPPO}

def main(args: Namespace) -> None:
    """
    Main function to run the RL algorithms based on the provided arguments.

    Args:
        args (Namespace): Command line arguments parsed by argparse.

    Raises:
        AssertionError: If time limit exceeds the configured maximum or if number of environments is less than 1.
        AssertionError: If the specified algorithm is not supported.
    """
    print("start main")
    max_time_limit = float(os.environ.get("MAX_TIME_LIMIT_MINUTES", "2800"))
    assert args.time_limit <= max_time_limit, (
        f"Invalid time limit: {args.time_limit}. Timeout limit is : {max_time_limit}"
    )
    start_time = time()
    
    # Update args with environment arguments
    args = ap.Namespace(**vars(args), **vars(get_env_args()))
    assert args.n_envs >= 1, f"Invalid n° of environments: {args.n_envs}. Must be >= 1"
    
    alg = args.alg.upper()
    assert alg in ALGORITHMS.keys(), f"Unsupported algorithm: {alg}. Supported algorithms are: {ALGORITHMS}"
    if args.distributed and alg != "MAPPO":
        raise NotImplementedError("Distributed multi-node training is implemented for MAPPO only.")
    if args.distributed and args.resume_run_name:
        raise NotImplementedError("Resuming distributed MAPPO runs is not implemented yet.")
    dist_info = init_distributed(args.dist_backend, args.dist_init_method) if args.distributed else None
    dist_rank = 0 if dist_info is None else dist_info.rank
    dist_world_size = 1 if dist_info is None else dist_info.world_size
    args.dist_rank = dist_rank
    args.dist_world_size = dist_world_size
    args.global_n_envs = args.n_envs * dist_world_size
    if (alg == "LAGRMAPPO" and args.constraints_type == 0) or (alg != "LAGRMAPPO" and args.constraints_type in [1, 2]):
        raise ValueError("Check the constrained version of the alg/env!")

    if args.resume_run_name:
        run_name = args.resume_run_name
    else:
        action_tag = "T" if args.action_type == "topology" else "R"
        heuristic_tag = "H" if args.use_heuristic else ""
        heuristic_type_tag = "I" if args.heuristic_type == "idle" else ""
        constraint_tag = "C1" if args.constraints_type == 1 else "C2" if args.constraints_type == 2 else ""
        run_name = (
            f"{args.alg}_{args.env_id}_{action_tag}_{args.seed}_{args.difficulty}_"
            f"{heuristic_tag}_{heuristic_type_tag}_{constraint_tag}_{int(time())}_"
            f"{np.random.randint(0, 50000)}"
        )

    # Initialize the appropriate checkpoint based on the algorithm
    if alg == 'MAPPO': checkpoint = MAPPOCheckpoint(run_name, args)
    elif alg == 'QPLEX': checkpoint = QPLEXCheckpoint(run_name, args)
    elif alg == 'LAGRMAPPO': checkpoint = LagrMAPPOCheckpoint(run_name, args)

    else:
        pass  # This case should not occur due to earlier assertion

    # Set random seed and Torch configuration
    set_random_seed(args.seed)
    set_torch(args.n_threads, args.th_deterministic, args.cuda)
    
    # Resume run if checkpoint was resumed
    if checkpoint.resumed: args = checkpoint.loaded_run['args']
  
    if args.distributed:
        print(
            f"Rank {dist_rank}/{dist_world_size}: creating {args.n_envs} local vector environments "
            f"({args.global_n_envs} global).",
            flush=True,
        )
    else:
        print(f"Creating {args.n_envs} vector environments...", flush=True)
    env_offset = dist_rank * args.n_envs
    env_fns = [lambda i=i: MAEnvWrapper(args, idx=env_offset + i) for i in range(args.n_envs)]
    envs = AsyncMultiAgentVecEnv(env_fns)

    try:
        # Run the specified algorithm
        ALGORITHMS[alg](envs, run_name, start_time, args, checkpoint)
    finally:
        if args.distributed:
            cleanup_distributed()
        
if __name__ == "__main__":
    # mp.get_context("forkserver")
    parser = ap.ArgumentParser()

    # Cluster
    parser.add_argument("--time-limit", type=float, default=1300, help="Time limit for the action ranking")
    parser.add_argument("--checkpoint", type=str2bool, default=False, help="Toggles checkpoint.")
    parser.add_argument("--resume-run-name", type=str, default='', help="Run name to resume")
    parser.add_argument("--distributed", type=str2bool, default=False, help="Enable Slurm/torch.distributed MAPPO training.")
    parser.add_argument("--dist-backend", type=str, default="gloo", help="torch.distributed backend.")
    parser.add_argument("--dist-init-method", type=str, default="env://", help="torch.distributed init method.")

    # Reproducibility [MAPPO, QPLEX, LAGRMAPPO]
    parser.add_argument("--alg", type=str, default='MAPPO', help="Algorithm to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Logger
    parser.add_argument("--verbose", type=str2bool, default=True, help="Toggles prints")
    parser.add_argument("--exp-tag", type=str, default='', help="Tag for logging the experiment")
    parser.add_argument("--track", type=str2bool, default=False, help="Tag for logging the experiment")
    parser.add_argument("--wandb-project", type=str, default="marl2grid_test", help="Wandb's project name.")
    parser.add_argument("--wandb-entity", type=str, default="emarche", help="Entity (team) of wandb's project.")
    parser.add_argument("--wandb-mode", type=str, default="online", help="Online or offline wandb mode.")

    # Torch
    parser.add_argument("--th-deterministic", type=str2bool, default=True, help="Enable deterministic in Torch.")
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    parser.add_argument("--n-threads", type=int, default=4, help="Max number of torch threads.")

    main(parser.parse_known_args()[0])
