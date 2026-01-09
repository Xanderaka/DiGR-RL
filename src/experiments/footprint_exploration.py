"""
Computational footprint exploration.

The purpose of this experiment is to explore which combination would have the best computational footprint and 
compare it to the F1 score, accuracy, and precision of the approach.

Research Questions:
    How can I best implement the code for the computational footprint?

Answers:
    Profile encoder, null_head, embed_head, and agent policy using torchprofile and fvcore.

Author:
    Xander Vreeswijk <xander.vreeswijk@xs4all.nl>
"""
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchprofile
from torch.utils.data import DataLoader

from fvcore.nn import FlopCountAnalysis

from src.data_utils.widar_dataset import WidarDataset
from src.utils.config_parser import parse_config_file
from src.data_utils.dataloader_collate import widar_collate_fn
from src.models.model_builder import build_model
from src.rl.rl_builder import build_rl
from src.rl.reward_functions import func_from_str
from stable_baselines3.common.base_class import BaseAlgorithm
import warnings

warnings.filterwarnings("ignore", message=r'No handlers found:.*', category=UserWarning)

# ----------------- Argument Parser -----------------
def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("checkpoint_fp", type=Path,
                   help="Path to the checkpoint")
    p.add_argument("agent_checkpoint_fp", type=Path,
                   help="Path to the agent checkpoint")
    p.add_argument("config_fp", type=Path,
                   help="Path to the config file")
    p.add_argument("data_dir_fp", type=Path,
                   help="Path to the data directory")
    return p.parse_args()

# ----------------- Wrappers -----------------
class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model(*inputs)

class HeadWrapper(nn.Module):
    """Wrapper for single-input heads / policy for FLOPs counting"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

# ----------------- Main Profiling Logic -----------------
def main(checkpoint_fp: Path, agent_checkpoint_fp: Path, config_fp: Path, data_dir_fp: Path):
    config = parse_config_file(config_fp)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load data
    bvp_pipeline = config["data"]["bvp_pipeline"]
    dataset_kwargs = {
        "dataset_type": config["data"]["dataset_type"],
        "return_bvp": True,
        "bvp_agg": config["data"]["bvp_agg"],
        "return_csi": not bvp_pipeline,
        "amp_pipeline": config["data"]["amp_pipeline"],
        "phase_pipeline": config["data"]["phase_pipeline"],
    }

    train_dataset = WidarDataset(data_dir_fp, "train", **dataset_kwargs)
    train_dataloader = DataLoader(train_dataset,
                                  config["train"]["batch_size"],
                                  num_workers=0,
                                  collate_fn=widar_collate_fn,
                                  drop_last=True,
                                  shuffle=True)

    reward_function = func_from_str(config["embed"]["reward_function"])
    encoder, null_head, null_agent = build_model(config, train_dataset)
    embed_head = copy.deepcopy(null_head)
    agent_type = None
    if agent_checkpoint_fp is not None:
        name_without_ext = os.path.splitext(agent_checkpoint_fp.name)[0]
        if name_without_ext.endswith("ppo"):
            agent_type = "ppo"
        elif name_without_ext.endswith("sac"):
            agent_type = "sac"
        else:
            raise ValueError(f"Unrecognized agent type from filename: {agent_checkpoint_fp}")

    env, embed_agent, _, _, _ = build_rl(
        encoder, null_head, embed_head, null_agent,
        agent_type, bvp_pipeline, device, train_dataloader,
        reward_function, 1, agent_type if agent_type else "none", False,
        config["embed"]["policy"], config
    )

    encoder.to(device)
    null_head.to(device)
    embed_head.to(device)
    embed_agent.policy.to(device)

    checkpoint = torch.load(checkpoint_fp, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    null_head.load_state_dict(checkpoint["null_mt_head_state_dict"])
    embed_head.load_state_dict(checkpoint["embed_mt_head_state_dict"])
    embed_agent.load(agent_checkpoint_fp)

    # ----------------- MACs -----------------
    total_encoder_macs = 0
    total_null_head_macs = 0
    total_embed_head_macs = 0
    total_agent_policy_macs = 0

    wrapped_encoder = EncoderWrapper(encoder).to(device).eval()

    with torch.no_grad():
        for amp, phase, bvp, info in tqdm(train_dataloader, desc="Profiling MACs"):
            amp, phase, bvp = amp.to(device), phase.to(device), bvp.to(device)
            inputs = (amp, phase, bvp)

            # Encoder
            macs_encoder = torchprofile.profile_macs(wrapped_encoder, inputs)
            total_encoder_macs += macs_encoder

            z = wrapped_encoder(*inputs)
            null_embedding = null_agent(z, info)
            if isinstance(embed_agent, BaseAlgorithm):
                obs = z.detach().cpu()
                action, _ = embed_agent.predict(obs)
                domain_embedding = torch.tensor(action, device=device, dtype=torch.float32)
                domain_embedding = F.softmax(domain_embedding, dim=1)
            else:
                raise ValueError("Unsupported embedding agent type")

            z_plus = torch.cat([z, domain_embedding], dim=1)

            # Heads and Agent
            macs_null = torchprofile.profile_macs(null_head, (z_plus,))
            macs_embed = torchprofile.profile_macs(embed_head, (z_plus,))
            macs_agent = torchprofile.profile_macs(embed_agent.policy, (z,))

            total_null_head_macs += macs_null
            total_embed_head_macs += macs_embed
            total_agent_policy_macs += macs_agent

    # ----------------- FLOPs (per forward pass with fvcore) -----------------
    print("\n📊 FLOPs per forward pass:")

    amp, phase, bvp, info = next(iter(train_dataloader))
    amp, phase, bvp = amp.to(device), phase.to(device), bvp.to(device)

    # Encoder FLOPs
    wrapped_encoder = EncoderWrapper(encoder).to(device)
    flops_encoder = FlopCountAnalysis(wrapped_encoder, (amp, phase, bvp))
    print(f"Encoder FLOPs: {flops_encoder.total()} ({flops_encoder.total()/1e9:.2f} GFLOPs)")

    z = encoder(amp, phase, bvp)
    obs = z.detach().cpu()
    action, _ = embed_agent.predict(obs)
    domain_embedding = torch.tensor(action, device=device, dtype=torch.float32)
    domain_embedding = F.softmax(domain_embedding, dim=1)
    z_plus = torch.cat([z, domain_embedding], dim=1)

    # Null Head FLOPs
    flops_null = FlopCountAnalysis(HeadWrapper(null_head).to(device), (z_plus,))
    print(f"Null Head FLOPs: {flops_null.total()} ({flops_null.total()/1e9:.2f} GFLOPs)")

    # Embed Head FLOPs
    flops_embed = FlopCountAnalysis(HeadWrapper(embed_head).to(device), (z_plus,))
    print(f"Embed Head FLOPs: {flops_embed.total()} ({flops_embed.total()/1e9:.2f} GFLOPs)")

    # Agent Policy FLOPs
    flops_policy = FlopCountAnalysis(HeadWrapper(embed_agent.policy).to(device), (z,))
    print(f"Agent Policy FLOPs: {flops_policy.total()} ({flops_policy.total()/1e9:.2f} GFLOPs)")

    # ----------------- Results -----------------
    print(f"\n📊 Total MACs across dataset:")
    print(f"Encoder MACs:        {total_encoder_macs:,}")
    print(f"Null Head MACs:      {total_null_head_macs:,}")
    print(f"Embed Head MACs:     {total_embed_head_macs:,}")
    print(f"Agent Policy MACs:   {total_agent_policy_macs:,}")
    print(f"✅ Overall Total:     {total_encoder_macs + total_null_head_macs + total_embed_head_macs + total_agent_policy_macs:,}")

# ----------------- Entry Point -----------------
if __name__ == '__main__':
    args = parse_args()
    main(args.checkpoint_fp, args.agent_checkpoint_fp,
         args.config_fp, args.data_dir_fp)
