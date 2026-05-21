import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch as th
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedInfo:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def _first_slurm_host() -> str:
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return "127.0.0.1"
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
        )
        return output.splitlines()[0].strip()
    except Exception:
        return "127.0.0.1"


def init_distributed(backend: str = "gloo", init_method: str = "env://") -> DistributedInfo:
    """Initialize torch.distributed from torchrun or Slurm environment variables."""
    if dist.is_initialized():
        return DistributedInfo(True, dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0)))

    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ.setdefault("MASTER_ADDR", _first_slurm_host())
    os.environ.setdefault("MASTER_PORT", "29500")

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Distributed mode needs RANK/WORLD_SIZE or Slurm's SLURM_PROCID/SLURM_NTASKS."
        )

    dist.init_process_group(backend=backend, init_method=init_method)
    return DistributedInfo(
        enabled=True,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )


def get_distributed_info() -> DistributedInfo:
    if not dist.is_available() or not dist.is_initialized():
        return DistributedInfo()
    return DistributedInfo(True, dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean_tensor(tensor: th.Tensor) -> th.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def all_reduce_mean_grads(parameters: Iterable[th.nn.Parameter]) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for param in parameters:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def normalize_advantages(advantages: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return (advantages - advantages.mean()) / (advantages.std() + eps)

    count = th.tensor([advantages.numel()], dtype=advantages.dtype, device=advantages.device)
    total = advantages.sum().reshape(1)
    total_sq = (advantages * advantages).sum().reshape(1)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
    mean = total / count
    var = th.clamp(total_sq / count - mean * mean, min=0.0)
    return (advantages - mean) / (th.sqrt(var) + eps)


def all_gather_1d_numpy(values: np.ndarray, dtype: th.dtype) -> np.ndarray:
    info = get_distributed_info()
    values = np.asarray(values)
    if not info.enabled:
        return values

    tensor = th.as_tensor(values, dtype=dtype)
    gathered = [th.empty_like(tensor) for _ in range(info.world_size)]
    dist.all_gather(gathered, tensor)
    return th.cat(gathered, dim=0).cpu().numpy()


def gather_normalized_rewards(reward_normalizer, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    info = get_distributed_info()
    if not info.enabled:
        return reward_normalizer(rewards, dones)

    global_rewards = all_gather_1d_numpy(np.asarray(rewards), th.float64)
    global_dones = all_gather_1d_numpy(np.asarray(dones).astype(np.int64), th.int64).astype(bool)
    normalized = reward_normalizer(global_rewards, global_dones)
    local_n = len(rewards)
    start = info.rank * local_n
    return normalized[start : start + local_n]


def merge_obs_stats(stats_list: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for stats in stats_list:
        for agent_id, s in stats.items():
            if s["mean"] is None:
                continue
            if agent_id not in merged:
                merged[agent_id] = {
                    "count": float(s["count"]),
                    "mean": s["mean"].copy(),
                    "var": s["var"].copy(),
                }
                continue
            a = merged[agent_id]
            n_a = a["count"]
            n_b = float(s["count"])
            n = n_a + n_b
            delta = s["mean"] - a["mean"]
            a["mean"] = a["mean"] + delta * (n_b / n)
            a["var"] = a["var"] + s["var"] + (delta**2) * (n_a * n_b / n)
            a["count"] = n
    return merged


def gather_obs_stats(local_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not dist.is_available() or not dist.is_initialized():
        return local_stats
    gathered: List[Dict[str, Dict[str, Any]]] = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_stats)
    return merge_obs_stats(gathered)
