import math
import numpy as np
import torch
import torch.nn.functional as F

from src.models.multi_task import MultiTaskHead, run_heads
from src.models.null_agent import NullAgent

def func_from_str(reward_func_name: str) -> callable:
    if reward_func_name == "environment_adaptation":
        return environment_adaptation_reward
    elif reward_func_name == "task_performance":
        return task_performance_reward
    elif reward_func_name == "exploration_driven":
        return exploration_driven_reward
    elif reward_func_name == 'do_nothing':
        return do_nothing
    elif reward_func_name == "combined":
        return combined_reward
    else:
        raise ValueError(f"Reward function given {reward_func_name} is not"
                         f"known.")

def do_nothing(null_head: MultiTaskHead,
                    embed_head: MultiTaskHead,
                    null_agent: NullAgent,
                    z: torch.Tensor,
                    obs: torch.Tensor,
                    info: dict[str, any],
                    action: torch.Tensor,
                    device: torch.device,):
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return -np.linalg.norm(action) * 0.1 + 0.5

def exploration_driven_reward(
        null_head, embed_head, null_agent,
        z: torch.Tensor, obs: torch.Tensor, info: dict,
        action: torch.Tensor, device: torch.device,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        entropy_bonus_weight: float = 0.1,
        distance_metric: str = "cosine") -> float:
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.float32, device=device)
    else:
        z = z.to(device)
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
    else:
        obs = obs.to(device)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype=torch.float32, device=device)
    else:
        action = action.to(device)

    z = _prepare_tensor(z, device)
    action = _prepare_tensor(action, device)
    null_embedding = null_agent(z, info).detach()
    
    with torch.no_grad():
        domain_null, gesture_null, _, _ = run_heads(
            null_head=null_head,
            embed_head=embed_head,
            null_embedding=null_embedding,
            agent_embedding=action,
            z=z
        )
    
    # Gesture cross-entropy loss -> score
    gesture_target = torch.tensor([info["gesture"]], device=device)
    gesture_loss = F.cross_entropy(gesture_null, gesture_target)
    gesture_score = torch.exp(-gesture_loss)

    # Domain accuracy
    domain_target = torch.tensor([info["domain_label"]], device=device)
    domain_pred = torch.argmax(domain_null, dim=1)
    domain_acc = (domain_pred == domain_target).float()

    # Distance novelty
    dist = _compute_distance(action, null_embedding, distance_metric)

    entropy_bonus = torch.tensor(0.0, device=device)
    if "policy_logits" in info:
        pi_logits = info["policy_logits"].to(device)
        pi_probs = F.softmax(pi_logits, dim=-1)
        entropy = -(pi_probs * torch.log(pi_probs + 1e-8)).sum(dim=-1).mean()
        entropy_bonus = entropy_bonus_weight * entropy

    reward = alpha * gesture_score + beta * domain_acc - gamma * dist + entropy_bonus

    if isinstance(reward, torch.Tensor):
        return reward.detach().cpu().item()
    else:
        return reward


def environment_adaptation_reward(
    null_head, embed_head, null_agent,
    z: torch.Tensor, obs: torch.Tensor, info: dict,
    action: torch.Tensor, device: torch.device,
    alpha: float = 1.0,
    beta: float = 0.5,
    lambda_: float = 0.5,
    distance_metric: str = "cosine") -> float:
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.float32, device=device)
    else:
        z = z.to(device)
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
    else:
        obs = obs.to(device)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype=torch.float32, device=device)
    else:
        action = action.to(device)

    z = _prepare_tensor(z, device)
    action = _prepare_tensor(action, device)
    null_embedding = null_agent(z, info).detach()

    _, gesture_null, domain_embed, _ = run_heads(
        null_head=null_head,
        embed_head=embed_head,
        null_embedding=null_embedding,
        agent_embedding=action,
        z=z
    )

    gesture_target = torch.tensor([info["gesture"]], device=device)
    gesture_loss = F.cross_entropy(gesture_null, gesture_target)
    gesture_score = torch.exp(-gesture_loss)

    domain_target = torch.tensor([info["domain_label"]], device=device)
    domain_preds = torch.argmax(domain_embed, dim=1)
    domain_acc = (domain_preds == domain_target).float().item()

    dist = _compute_distance(action, null_embedding, distance_metric)

    # Here, lower domain entropy means more confident domain prediction,
    # so reward encourages higher entropy (domain confusion) for adaptation
    reward = alpha * gesture_score + beta * domain_acc - lambda_ * dist
    if isinstance(reward, torch.Tensor):
        return reward.detach().cpu().item()
    else:
        return reward


def task_performance_reward(
    null_head, embed_head, null_agent,
    z: torch.Tensor, obs: torch.Tensor, info: dict,
    action: torch.Tensor, device: torch.device,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    distance_metric: str = "cosine") -> float:
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.float32, device=device)
    else:
        z = z.to(device)
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
    else:
        obs = obs.to(device)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype=torch.float32, device=device)
    else:
        action = action.to(device)
    z = _prepare_tensor(z, device)
    action = _prepare_tensor(action, device)
    null_embedding = null_agent(z, info).detach()

    with torch.no_grad():
        domain_null, gesture_null, domain_embed, gesture_embed = run_heads(
            null_head=null_head,
            embed_head=embed_head,
            null_embedding=null_embedding,
            agent_embedding=action,
            z=z
        )

    gesture_target = torch.tensor([info["gesture"]], device=device)
    gesture_loss = F.cross_entropy(gesture_null, gesture_target)
    gesture_score = torch.exp(-gesture_loss)

    domain_probs = F.softmax(domain_null, dim=1)
    domain_confidence = domain_probs.max(dim=1).values.mean()

    dist = _compute_distance(action, null_embedding, distance_metric)

    # Reward encourages gesture score, penalizes domain confidence and embedding distance
    reward = alpha * gesture_score - beta * domain_confidence - gamma * dist
    if isinstance(reward, torch.Tensor):
        return reward.detach().cpu().item()
    else:
        return reward


def combined_reward(
    null_head, embed_head, null_agent,
    z: torch.Tensor, obs: torch.Tensor, info: dict,
    action: torch.Tensor, device: torch.device,
    w_env: float = 1.0,
    w_task: float = 1.0,
    w_explore: float = 1.0,
    env_params: dict = None,
    task_params: dict = None,
    explore_params: dict = None
) -> float:
    """
    Combines environment adaptation, task performance, and exploration-driven rewards.
    Each component is weighted by w_env, w_task, w_explore respectively.
    """
    if env_params is None:
        env_params = {}
    if task_params is None:
        task_params = {}
    if explore_params is None:
        explore_params = {}

    r_env = environment_adaptation_reward(
        null_head, embed_head, null_agent, z, obs, info, action, device, **env_params
    )
    r_task = task_performance_reward(
        null_head, embed_head, null_agent, z, obs, info, action, device, **task_params
    )
    r_explore = exploration_driven_reward(
        null_head, embed_head, null_agent, z, obs, info, action, device, **explore_params
    )

    combined = w_env * r_env + w_task * r_task + w_explore * r_explore

    return combined

# Helper functions
def _prepare_tensor(x, device):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def _compute_distance(a, b, metric="cosine"):
    if metric == "cosine":
        dist = 1.0 - F.cosine_similarity(a, b).mean()
        dist = dist / 2.0  # normalize to [0,1]
    elif metric == "mse":
        dist = F.mse_loss(a, b)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return dist
