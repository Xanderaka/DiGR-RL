import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from gymnasium import spaces

from src.models.null_agent import NullAgent


class LatentEnvironment(gym.Env):
    metadata = {"render_modes": ["none"]}

    def __init__(self,
                 encoder: nn.Module,
                 null_head: nn.Module,
                 embed_head: nn.Module,
                 null_agent: NullAgent,
                 device: torch.device,
                 dataset: Dataset,
                 reward_function: callable):
        self.encoder = encoder
        self.null_head = null_head
        self.embed_head = embed_head
        self.null_agent = null_agent
        self.dataset = dataset
        self.device = device
        self.reward_function = reward_function

        self.last_obs = None
        self.last_z = None
        self.last_info = None

        self.current_step = 0
        self.max_episode_steps = len(dataset)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=[2 * self.encoder.latent_dim]
        )

        # Action space corresponds to null_head domain labels
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=[self.null_head.domain_label_size]
        )

        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _set_obs(self, step_num):
        """Extracts data from dataset and encodes into latent representation."""
        amp, phase, bvp, info = self.dataset[step_num]

        # Move inputs to device and add batch dimension)
        amp = amp.to(self.device).unsqueeze(0)
        phase = phase.to(self.device).unsqueeze(0)
        bvp = bvp.to(self.device).unsqueeze(0)

        with torch.no_grad():
            z = self.encoder(amp, phase, bvp)

        if not isinstance(info, dict):
            info = {"info": info}

        self.last_z = z.detach().flatten().unsqueeze(0)
        self.last_obs = self.last_z.cpu()
        self.last_info = info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Save previous episode reward if any
        if self.current_episode_reward > 0.0:
            self.episode_rewards.append(self.current_episode_reward)

        self.current_step = 0
        self.current_episode_reward = 0.0

        self._set_obs(0)
        return self.last_obs, self.last_info

    def step(self, action):
        terminated = self.current_step >= self.max_episode_steps - 1

        # Ensure action is a tensor with batch dimension
        if isinstance(action, np.ndarray):
            action = torch.tensor(action).to(self.device)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        reward = self.reward_function(
            self.null_head,
            self.embed_head,
            self.null_agent,
            self.last_z,
            self.last_obs,
            self.last_info,
            action,
            self.device
        )
        self.current_episode_reward += reward

        # Advance to next observation if episode not terminated
        if not terminated:
            self.current_step += 1
            self._set_obs(self.current_step)

        if terminated:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        truncated = False  # TimeLimit wrapper handles truncation
        return self.last_obs, reward, terminated, truncated, self.last_info

    # Convenience functions to access rewards
    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_reward_mean(self):
        return np.mean(self.episode_rewards) if self.episode_rewards else float("-inf")
