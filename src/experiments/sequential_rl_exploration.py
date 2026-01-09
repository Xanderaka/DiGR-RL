"""Sequential reinforcement learning exploration.

The purpose of this experiment is to explore how to best integrate the sequential reinforcement
learning approach and how to switch between the two rl agents.

Author:
    Xander Vreeswijk <xander.vreeswijk@xs4all.nl>
"""
import os
import copy
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.data_utils.widar_dataset import WidarDataset
from src.models.model_builder import build_model
from rl.rl_builder import build_rl
from src.rl.reward_functions_new import func_from_str
from stable_baselines3 import DDPG, PPO, SAC

from src.utils.config_parser import parse_config_file

from stable_baselines3.common.callbacks import BaseCallback

import time

class EPRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rew_mean = None  # This will store the most recent value

    def _on_step(self) -> bool:
        # Get the episode info buffer
        ep_infos = self.model.ep_info_buffer

        if len(ep_infos) > 0:
            # Extract 'r' keys and compute safe mean
            ep_rewards = [ep_info['r'] for ep_info in ep_infos if 'r' in ep_info]
            self.ep_rew_mean = np.mean(ep_rewards) if ep_rewards else float('nan')

        return True

def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("config_fp", type=Path,
                   help="Path to the config file")
    p.add_argument("data_dir_fp", type=Path,
                   help="Path to the data directory")

    return p.parse_args()

def get_episode_rewards_from_env(env):
    # Try direct method from environment (e.g., custom method)
    if hasattr(env, 'get_episode_rewards'):
        return env.get_episode_rewards()
    # Try unwrapping TimeLimit or other wrappers
    elif hasattr(env, 'env') and hasattr(env.env, 'get_episode_rewards'):
        return env.env.get_episode_rewards()
    # Try reading from SB3's Monitor wrapper
    elif hasattr(env, 'episode_rewards'):
        return env.episode_rewards
    elif hasattr(env, 'env') and hasattr(env.env, 'episode_rewards'):
        return env.env.episode_rewards
    else:
        return []

def main(config_fp: Path, data_dir_fp: Path):
    config = parse_config_file(config_fp)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    bvp_pipeline = config["data"]["bvp_pipeline"]
    train_dataset = WidarDataset(data_dir_fp, "train",
                                 config["data"]["dataset_type"],
                                 return_bvp=True,
                                 bvp_agg=config["data"]["bvp_agg"],
                                 return_csi=not bvp_pipeline,
                                 amp_pipeline=config["data"]["amp_pipeline"],
                                 phase_pipeline=config["data"][
                                     "phase_pipeline"])
    valid_dataset = WidarDataset(data_dir_fp, "validation",
                                 config["data"]["dataset_type"],
                                 return_bvp=True,
                                 bvp_agg=config["data"]["bvp_agg"],
                                 return_csi=not bvp_pipeline,
                                 amp_pipeline=config["data"]["amp_pipeline"],
                                 phase_pipeline=config["data"][
                                     "phase_pipeline"])
    train_loader = DataLoader(train_dataset)

    # SECTION Model
    reward_function = func_from_str(config["embed"]["reward_function"])
    policy_name = config["embed"]["policy"]
    embed_agent = config["embed"]["agent_type"]
    encoder, null_head, null_agent = build_model(config, train_dataset)

    recent_rewards = []
    threshold = 50
    switch_to_sac = False
    embed_agent_type = "ppo"
    reward_stability_window = 5
    embed_head = copy.deepcopy(null_head)
    for epoch in range(15):
        print(f"\n🌀 Epoch {epoch} --------------------------")

        rl = build_rl(encoder, null_head, embed_head, null_agent,
                                    embed_agent,
                                    bvp_pipeline, device, train_loader,
                                    reward_function, 1, embed_agent_type, switch_to_sac, policy_name)

        env, embed_agent, steps, embed_agent_type, switch_to_sac = rl
        encoder.eval()
        null_head.eval()
        embed_head.eval()

        # Create and train model
        callback = EPRewardLogger(verbose=1)
        embed_agent.learn(total_timesteps=steps, callback=callback)

        # Access the ep_rew_mean after training
        ep_rew_mean = callback.ep_rew_mean
        print("Final ep_rew_mean:", callback.ep_rew_mean)

        # Track reward deltas
        if embed_agent_type == "ppo":
            recent_rewards.append(ep_rew_mean)
            if len(recent_rewards) >= reward_stability_window:
                reward_std = np.std(recent_rewards[-reward_stability_window:])
                print(f"📊 Reward std over last {reward_stability_window} epochs: {reward_std:.2f}")

                if reward_std < threshold:
                    print("✅ Reward stable. Switching to SAC next epoch.")
                    switch_to_sac = True
    return 1

if __name__ == '__main__':
    args = parse_args()
    main(args.config_fp, args.data_dir_fp)