"""Train Runner.

Provides all necessary objects and parameters to make training possible.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser
from pathlib import Path
import os
import warnings
import time
import sys

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import wandb
from torch.utils.data import random_split


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_utils.widar_dataset import WidarDataset
from src.data_utils.dataloader_collate import widar_collate_fn
from src.utils.config_parser import parse_config_file
from src.experiments.train import Training
from src.loss.multi_joint_loss import MultiJointLoss
from src.models.model_builder import build_model
from src.rl.reward_functions import func_from_str
from src.ui.tqdm_ui import TqdmUI
from src.ui.cluster_logger import ClusterLoggerUI

def parse_args():
    p = ArgumentParser(description="Runner for DARLInG Training")
    p.add_argument("CONFIG_FP", type=Path,
                   help="Path to the config yaml file.")
    return p.parse_args()


CONV_NUM_FEATURES_MAPS = {
    "bvp_stack": 4608,     # 28x20x20 dimensional BVP
    "bvp_1d": 102400,      # 1x28x400 dimensional BVP
    "bvp_sum": 4608,       # 1x20x20 dimensional BVP
    "sti_transform": 0     # 540x1024x1024 dimensional signal-to-image transform
}


def run_training(config: dict[str, dict[str, any]]):
    """Runs the training.

    """
    # SECTION Initial stuff
    # Set tags
    torch.autograd.set_detect_anomaly(True)
    if config["data"]["transformation"] is None:
        transformation = "no_transform"
    else:
        transformation = str(config["data"]["transformation"])
    Backbone = config["encoder"]["Backbone_type"]
    tags = [transformation, Backbone, "training"]
    if config["debug"]["is_debug"]:
        warnings.warn("Running a debug run, `debug` will be appended to tags.")
        tags.append("debug")
    # Init wandb
    if config["debug"]["offline"]:
        os.environ["WANDB_MODE"] = "dryrun"
        warnings.warn("Running WandB in offline mode.")
    run = wandb.init(project="master-thesis-test", entity="x-j-vreeswijk-eindhoven-university-of-technology",
                     config=config, tags=tags)
    # Validate checkpoints dir
    if not config["train"]["checkpoint_dir"].exists():
        config["train"]["checkpoint_dir"].mkdir(parents=True)

    # Set device
    if config["debug"]["on_cpu"]:
        # Debugging on CPU is easier
        print("Running training on CPU")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("Running training on CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Running training on MPS")
        device = torch.device("mps")
    else:
        print("Running training on CPU")
        device = torch.device("cpu")

    # SECTION Data
    data_dir = config["data"]["data_dir"]
    bvp_agg = config["data"]["bvp_agg"]
    dataset_type = config["data"]["dataset_type"]
    train_dataset = WidarDataset(
        data_dir,
        "train",
        dataset_type,
        amp_pipeline=config["data"]["amp_pipeline"],
        phase_pipeline=config["data"]["phase_pipeline"],
        return_csi=True,
        return_bvp=True,
        bvp_agg=bvp_agg
    )
    valid_dataset = WidarDataset(
        data_dir,
        "validation",
        dataset_type,
        amp_pipeline=config["data"]["amp_pipeline"],
        phase_pipeline=config["data"]["phase_pipeline"],
        return_csi=True,
        return_bvp=True,
        bvp_agg=bvp_agg
    )
    # Use less of the dataset for hyperparamater tuning
    if config["train"]["tuning"]:
        print("Trimming the dataset for tuning")
        total_size = len(train_dataset)
        test_size = int(0.2 * total_size)
        train_size = total_size - test_size

        _, train_subset = random_split(
            train_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Suppose it's (x, y, domain_idx) → domain_idx = sample[2]
        domains = set()
        for idx in train_subset.indices:
            _, _, _, info = train_subset.dataset[idx]
            domains.add(info["domain_idx"])

        domain_to_idx = {d: i for i, d in enumerate(sorted(domains))}
        train_subset.domain_to_idx = domain_to_idx

        train_dataset = train_subset
    # 1 worker ensure no multithreading so we can debug easily
    # num_workers = 1 if is_debug else (torch.get_num_threads() - 2) // 2
    # Trying to get a process lock for the dataloader takes way too long if
    # num_workers > 0 (~3x longer) so we set num_workers to always be 0
    num_workers = 0
    train_dataloader = DataLoader(train_dataset,
                                  config["train"]["batch_size"],
                                  num_workers=num_workers,
                                  collate_fn=widar_collate_fn,
                                  drop_last=True,
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  config["train"]["batch_size"],
                                  num_workers=num_workers,
                                  collate_fn=widar_collate_fn,
                                  drop_last=True)

    # SECTION Set up models
    encoder, null_head, null_agent = build_model(
        config,
        train_dataset
    )
    embed_agent = config["embed"]["agent_type"]

    # Move models to device
    encoder.to(device)
    null_head.to(device)
    null_agent.to(device)
    # Loss and optimizers
    optimizer = config["optim_loss"]["optimizer"]
    lr = config["optim_loss"]["lr"]
    loss_fn = MultiJointLoss(config["optim_loss"]["alpha"])
    optimizer_map = {"adam": Adam, "sgd": SGD}

    encoder_optimizer = optimizer_map[optimizer](encoder.parameters(), lr=lr)
    null_optimizer = optimizer_map[optimizer](null_head.parameters(), lr=lr)

    # SECTION Reward function
    reward_function = func_from_str(config["embed"]["reward_function"])
    policy_name = config["embed"]["policy"]

    # SECTION UI
    initial_data = {"train_loss": float("nan"),
                    "train_kl_loss": float("nan"),
                    "train_null_loss": float("nan"),
                    "train_embed_loss": float("nan"),
                    "valid_loss": float("nan"),
                    "loss_diff": float("nan"),
                    "epoch": "0",
                    "batch": "0",
                    "rate": float("nan")}
    train_steps = len(train_dataset)
    match config["train"]["ui"]:
        case "tqdm":
            ui = TqdmUI(train_steps, len(valid_dataset),
                        config["train"]["epochs"], initial_data)
        case "gui":
            raise NotImplementedError("GUI has not been implemented yet.")
        case "cluster_ui":
            ui = ClusterLoggerUI(train_steps, len(valid_dataset),
                                 config["train"]["epochs"], initial_data)
        case _:
            raise ValueError(f"{config['train']['ui']} is not one of the "
                             f"available options."
                             f"Available options are [`tqdm`, `cluster_ui`]")

    ui.update_status("Preparation complete. Starting training...")

    # SECTION Run training
    checkpoint_dir = config["train"]["checkpoint_dir"]
    training = Training(
        encoder, null_head,                                # Models
        embed_agent, null_agent,                           # Embed agents
        encoder_optimizer, null_optimizer,                 # Optimizers
        loss_fn,                                           # Loss function
        run, checkpoint_dir, ui,                           # Utils
        optimizer_map[optimizer], lr,                      # Optimizer
        reward_function,                                   # Reward function
        policy_name,                                       # Policy
        config,
        agent_start_epoch=config["embed"]["start_epoch"],  # Embed config
    )
    best_loss = training.train(
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        epochs=config["train"]["epochs"],
        device=device,
        agent_epochs=config["embed"]["epochs"]
    )
    return best_loss


if __name__ == '__main__':
    args = parse_args()
    print("Running from config file:")
    print(f"{args.CONFIG_FP}")
    best_loss = run_training(parse_config_file(args.CONFIG_FP))
