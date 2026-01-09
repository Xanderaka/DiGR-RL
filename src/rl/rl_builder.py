import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.ppo.policies import ActorCriticPolicy
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from src.models.null_agent import NullAgent
from src.rl.latent_environment import LatentEnvironment
from src.rl.custom_ppo_policy import CustomPPOPolicy
from src.rl.custom_sac_policy import CustomSACPolicy
from src.rl.Policies import EBEPPolicyPPO, EBEPPolicySAC, EDBPolicyPPO, EDBPolicySAC, AEEPPolicyPPO, AEEPPolicySAC

# === Map string identifiers to SAC/PPO policy classes
def pol_from_str(policy_name: str):
    if policy_name == "exploration":
        return EBEPPolicySAC, EBEPPolicyPPO
    elif policy_name == "exploitation":
        return EDBPolicySAC, EDBPolicyPPO
    elif policy_name == "hybrid":
        return AEEPPolicySAC, AEEPPolicyPPO
    elif policy_name == 'do_nothing':
        return CustomSACPolicy, CustomPPOPolicy
    else:
        raise ValueError(f"Policy given {policy_name} is not known.")


def build_rl(
    encoder: nn.Module,
    null_head: nn.Module,
    embed_head: nn.Module,
    null_agent: NullAgent,
    embedding_agent: str,
    device: torch.device,
    train_loader: DataLoader,
    reward_function: callable,
    agent_epochs: int,
    embed_agent_type: str,
    switch_to_sac: bool,
    policy_name:str,
    config:dict[str, dict[str, any]]
) -> tuple:
    """Build latent environment and RL agent (PPO or SAC).

    Returns:
        env: LatentEnvironment wrapped with Monitor and TimeLimit
        embedding_agent: RL agent (PPO or SAC)
        total_agent_timesteps: total steps allocated for training
        embed_agent_type: "ppo" or "sac"
        switch_to_sac: flag indicating if agent was switched to SAC
    """
    # === Environment Setup ===
    env = LatentEnvironment(
        encoder=encoder,
        null_head=null_head,
        embed_head=embed_head,
        null_agent=null_agent,
        device=device,
        dataset=train_loader.dataset,
        reward_function=reward_function,
    )
    env = TimeLimit(env, max_episode_steps=200)  # Cap episode length
    env = Monitor(env)  # Record episode statistics

    total_agent_timesteps = agent_epochs
    sac_policy, ppo_policy = pol_from_str(policy_name)

    # === PPO Agent Setup ===
    if embedding_agent == "ppo":
        embed_agent_type = "ppo"
        embedding_agent = PPO(
            ppo_policy,
            env,
            device=device,
            verbose=0,
            n_epochs=10,
            n_steps=4096,
            batch_size=512
        )
        total_agent_timesteps *= len(train_loader.dataset)  # Scale timesteps by dataset size

    # === SAC Agent Setup ===
    elif embedding_agent == "sac":
        embed_agent_type = "sac"
        embedding_agent = SAC(
            sac_policy,
            env,
            device=device,
            verbose=0,
            ent_coef=config["agent"]["ent_coef"],
            learning_rate=config["agent"]["learning_rate"],
            buffer_size=config["agent"]["buffer_size"],
            train_freq=(128, "step"),
            gradient_steps=config["agent"]["gradient_steps"],
            batch_size=config["agent"]["batch_size"],
            learning_starts=config["agent"]["learning_starts"],
        )

    # === PPO → SAC Transition with Soft Action Blending ===
    if switch_to_sac:
        switch_to_sac = False
        ppo_agent = embedding_agent
        ppo_policy_state_dict = ppo_agent.policy.state_dict()

        # Initialize SAC agent and load PPO weights partially
        embedding_agent = SAC(
            sac_policy,
            env,
            device=device,
            verbose=0,
            ent_coef=config["agent"]["ent_coef"],
            learning_rate=config["agent"]["learning_rate"],
            buffer_size=config["agent"]["buffer_size"],
            train_freq=(128, "step"),
            gradient_steps=config["agent"]["gradient_steps"],
            batch_size=config["agent"]["batch_size"],
            learning_starts=config["agent"]["learning_starts"],
        )
        embed_agent_type = "sac"
        embedding_agent.policy.load_state_dict(ppo_policy_state_dict, strict=False)

        # Bootstrap replay buffer by blending PPO and SAC actions
        obs, _ = env.reset()
        n_bootstrap_steps = int(1.5 * config["agent"]["learning_starts"])

        for step in range(n_bootstrap_steps):
            alpha = step / n_bootstrap_steps  # Linear blending factor

            action_ppo, _ = ppo_agent.predict(obs, deterministic=False)
            action_sac, _ = embedding_agent.predict(obs, deterministic=False)

            # Soft blending: gradually transition from PPO to SAC
            action = alpha * action_sac + (1 - alpha) * action_ppo

            # Step environment
            next_obs, reward, terminated, truncated, infos = env.step(action)
            done = terminated or truncated

            # Ensure infos is a list of dicts (required by replay buffer)
            if isinstance(infos, dict):
                infos = [infos]
            elif isinstance(infos, list) and isinstance(infos[0], dict):
                pass
            else:
                raise ValueError(f"Unexpected format for infos: {type(infos)}")

            # Add to SAC replay buffer
            embedding_agent.replay_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=infos
            )

            obs = next_obs
            if done:
                obs, _ = env.reset()

    # === Adjust total timesteps for SAC based on dataset and factor
    if isinstance(embedding_agent, SAC):
        dataset_size = len(train_loader.dataset)
        factor = config["agent"]["factor"]
        total_agent_timesteps *= dataset_size * factor

    return env, embedding_agent, total_agent_timesteps, embed_agent_type, switch_to_sac
