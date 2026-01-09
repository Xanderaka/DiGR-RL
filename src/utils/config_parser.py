import warnings
from pathlib import Path
from typing import Optional, List

import yaml
import time

# from signal_to_image.deepinsight_transform import DeepInsight
from src.signal_to_image.gaf_transform import GAF
from src.signal_to_image.mtf_transform import MTF
from src.signal_to_image.recurrence_plot_transform import RP
from src.signal_processing.pipeline import Pipeline
from src.signal_processing.standard_scaler import StandardScaler


def train_config(batch_size: int = 32,
                 epochs: int = 15,
                 ui: str = "tqdm",
                 checkpoint_dir: str | Path = Path("../../checkpoints/"),
                 tuning: bool = False
                 ) -> dict[str, any]:
    """Training configuration.

    A function is used to ensure defaults and clear documentation of possible
    options.

    Args:
        batch_size: The batch size to use for training.
        epochs: Number of epochs to train for.
        ui: UI type to use during training.
        checkpoint_dir: Where to save checkpoints to.
    """
    if type(checkpoint_dir) is str:
        checkpoint_dir = Path(checkpoint_dir)

    return {"batch_size": batch_size,
            "epochs": epochs,
            "ui": ui,
            "checkpoint_dir": checkpoint_dir,
            "tuning": tuning}


def data_config(data_dir: str | Path = Path("../../data/"),
                dataset_type: str = None,
                downsample_multiplier: int = 2,
                transformation: str = None,
                bvp_agg: str = "stack",
                amp_pipeline: Optional[list[str]] = None,
                phase_pipeline: Optional[list[str]] = None) -> dict[str, any]:
    """Data configuration for training.

    A function is used to ensure defaults and clear documentation of possible
    options.

    Args:
        data_dir: Path to the data directory.
        dataset_type: Type of the dataset. Options are [`small`,
                `single_domain`, `single_user`, `full`, `single_domain_small`,
                `single_user_small`].
        downsample_multiplier: How much to downsample the data by.
        transformation: Transformation to use on the data. If None, no
            transformation is used.
        bvp_agg: How to aggregate the BVP data. Options are
            [`stack`, `1d`, `sum`].
        amp_pipeline: Pipeline to use for the amplitude data. This is provided
            as a list of strings, where each string is a function to call on
            the data. The possible functions are:
                [`lowpass_filter`, `phase_derivative`, `phase_filter`,
                `phase_unwrap`, `standard_scalar`, `torch.from_numpy`].
            The default is [`torch.from_numpy`].
        phase_pipeline: Pipeline to use for the phase data. This is provided
            as a list of strings, where each string is a function to call on
            the data. The possible functions are:
                [`lowpass_filter`, `phase_derivative`, `phase_filter`,
                `phase_unwrap`, `standard_scalar`, `torch.from_numpy`].
            The default is [`torch.from_numpy`].
    """
    if type(data_dir) is str:
        data_dir = Path(data_dir)

    if dataset_type is None:
        raise ValueError("`dataset_type` parameter must be filled.")

    # Set the signal to image transformation to use.
    match transformation:
        case "deepinsight":
            raise NotImplementedError
            # transform = DeepInsight()
        case "gaf":
            transform = GAF()
        case "mtf":
            transform = MTF(image_size=1.)
        case "rp":
            transform = RP(time_delay=1)
        case None:
            transform = None
        case _:
            raise ValueError(
                f"Chosen transformation {transformation} is not one of the "
                f"valid options. Valid options are "
                f"[`deepinsight`, `gaf`, `mtf`, `rp`]"
            )

    if amp_pipeline is None:
        amp_pipeline = ["torch.from_numpy"]
    else:
        amp_pipeline = Pipeline.from_str_list(
            amp_pipeline,
            transform,
            StandardScaler(data_dir, "amp"),
            downsample_multiplier
        )
    if phase_pipeline is None:
        phase_pipeline = ["torch.from_numpy"]
    else:
        phase_pipeline = Pipeline.from_str_list(
            phase_pipeline,
            transform,
            StandardScaler(data_dir, "phase"),
            downsample_multiplier
        )

    return {"data_dir": data_dir,
            "dataset_type": dataset_type,
            "downsample_multiplier": downsample_multiplier,
            "transformation": transform,
            "bvp_agg": bvp_agg,
            "amp_pipeline": amp_pipeline,
            "phase_pipeline": phase_pipeline}



def encoder_config(dropout: float = 0.3,
                   latent_dim: int = 10,
                   activation_fn: str = "relu",
                   type: str = "DenseNet",
                   initial_kernel_size: int = 3,
                   normalization: bool = False,
                   init_channels: int = 64,
                   growth_rate: int = 32,
                   base_channels: int = 128,
                   patch_size: int = 64,
                   vit_depth: int = 6,
                   vit_heads: int = 4,
                   num_conv_layers: int = None,
                   layers_per_stage: Optional[List[int]]= [2,2,2,2]) -> dict[str, any]:
    """Encoder configuration for training.

    Args:
        dropout: Dropout rate to use.
        latent_dim: Dimension of the latent space.
        activation_fn: Activation function to use. Options are
            [`relu`, `leaky`, `selu`].
        type: type of backbone architecture. Options are
            ['DenseNet', 'ResNet', 'ViT'].
        initial_kernel_size: Initial kernel size to use for the convolutional
            layers.
    """
    return {"dropout": dropout,
            "latent_dim": latent_dim,
            "activation_fn": activation_fn,
            "Backbone_type": type,
            "normalization": normalization,
            "initial_kernel_size": initial_kernel_size,
            "init_channels": init_channels,
            "growth_rate": growth_rate,
            "base_channels": base_channels,
            "patch_size": patch_size,
            "vit_depth": vit_depth,
            "vit_heads": vit_heads,
            "num_conv_layers": num_conv_layers,
            "layers_per_stage": layers_per_stage}


def mt_config(d_predictor_dropout: float = 0.3,
              d_predictor_activation_fn: str = "selu",
              d_predictor_num_layers: int = 3,
              predictor_dropout: float = 0.3,
              predictor_activation_fn: str = "selu",
              predictor_num_layers: int = 3) -> dict[str, any]:
    """Multitask configuration for training.

    Args:
        decoder_dropout: Dropout rate to use for the decoder.
        decoder_activation_fn: Activation function to use for the decoder.
            Options are [`relu`, `leaky`, `selu`].
        predictor_dropout: Dropout rate to use for the predictor.
        predictor_activation_fn: Activation function to use for the predictor.
            Options are [`relu`, `leaky`, `selu`].
        predictor_num_layers: Number of layers to use for the GesturePredictor.
    """
    return {"d_predictor_dropout": d_predictor_dropout,
            "d_predictor_activation_fn": d_predictor_activation_fn,
            "d_predictor_num_layers": d_predictor_num_layers,
            "predictor_dropout": predictor_dropout,
            "predictor_activation_fn": predictor_activation_fn,
            "predictor_num_layers": predictor_num_layers}


def agent_config(ent_coef: float = 0.1,
                 learning_rate: float = 3e-4,
                 buffer_size: int = 50000,
                 train_freq: list = [64, "step"],
                 gradient_steps: int = 1,
                 batch_size: int = 64,
                 learning_starts: int = 1000,
                 factor: int = 2,
                 threshold: int = 5,
                 window: int = 20,
                 switch_epoch: int = 20,
                 sequential: bool = True) -> dict[str, any]:
    """Agent configuration for reinforcement learning.

    Args:
        ent_coef: Entropy coefficient for exploration.
        learning_rate: Learning rate for the agent optimizer.
        buffer_size: Size of the replay buffer.
        train_freq: Training frequency (steps).
        gradient_steps: Number of gradient updates after each rollout.
        batch_size: Mini-batch size for training.
        learning_starts: Number of steps before training begins.
        factor: Factor that will be multiplied with PPO training steps for SAC training
        threshold: Threshold used for adaptive switching to SAC from PPO
        window: Number of rewards the check will look over
        switch_epochs: after how many epochs the check for the switch will happen
        sequential: The check if RL will be sequential of single RL
    """
    return {"ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "factor": factor,
            "threshold": threshold,
            "window": window,
            "switch_epoch": switch_epoch,
            "sequential": sequential}

def embed_config(agent_type: str = "ppo",
                 reward_function: str = "combined",
                 policy: str = "do_nothing",
                 embed_size: Optional[int] = None,
                 start_epoch: int = 0,
                 epochs: int = 1) -> dict[str, any]:
    """Embedding agent configuration for training.

    Args:
        agent_type: Agent type. Options are
            [`sac`, `ppo`]
        reward_function: The reward function to use. Options are
            [`environment_adaption`, `task_performance`, `exploration_driven`,`combined`]
        policy: The policy strategy to use. Options are
            [`exploration`, `exploitation`, `hybrid`, `do_nothing`]
        embed_size: Size of the embedding. None is only allowed if the
            embed_agent_value is `known` and is automatically replaced by 33.
        epochs: Number of epochs to train the embedding agent for.
        start_epoch: Start training the agent after how many epochs. This is
            because the output of the decoder may not make sense before a few
            epochs of training and it does not make sense to train the agent
            on nonsensical inputs.
    """
    if embed_size is None:
        embed_size = 33
    return {"agent_type": agent_type,
            "reward_function": reward_function,
            "policy": policy,
            "embed_size": embed_size,
            "start_epoch": start_epoch,
            "epochs": epochs}

def optim_loss_config(optimizer: str = "adam",
                      lr: float = 0.001,
                      alpha: float = 0.5,
                      beta: float = 0.5) -> dict[str, any]:
    """Optimizer and loss configuration for training.

    Args:
        optimizer: Optimizer to use. Options are [`adam`, `sgd`].
        lr: Learning rate to use.
        alpha: Weight for the reconstruction vs classification loss.
        beta: Weight for the KL divergence vs mt_head loss.
    """
    return {"optimizer": optimizer,
            "lr": lr,
            "alpha": alpha,
            "beta": beta}


def debug_config(is_debug: bool = False,
                 on_cpu: bool = False,
                 offline: bool = False):
    """Debug configuration for training.

    Args:
        is_debug: Whether to run in debug mode.
        on_cpu: Whether to force running on the CPU.
        offline: Whether to run wandb in offline mode.
    """
    return {"is_debug": is_debug,
            "on_cpu": on_cpu,
            "offline": offline}


def parse_config_file(config_fp: Path) -> dict[str, dict[str, any]]:
    """Parses the yaml config file."""
    with open(config_fp, "r") as f:
        yaml_dict = yaml.safe_load(f)

    config_dict = {}

    if "train" in yaml_dict:
        config_dict["train"] = train_config(**yaml_dict["train"])
    else:
        config_dict["training"] = train_config()
    if "data" in yaml_dict:
        config_dict["data"] = data_config(**yaml_dict["data"])
    else:
        config_dict["data"] = data_config()
    if "encoder" in yaml_dict:
        config_dict["encoder"] = encoder_config(**yaml_dict["encoder"])
    else:
        config_dict["encoder"] = encoder_config()
    if "mt" in yaml_dict:
        config_dict["mt"] = mt_config(**yaml_dict["mt"])
    else:
        config_dict["mt"] = mt_config()
    if "agent" in yaml_dict:
        config_dict["agent"] = agent_config(**yaml_dict["agent"])
    else:
        config_dict["agent"] = agent_config()
    if "embed" in yaml_dict:
        config_dict["embed"] = embed_config(**yaml_dict["embed"])
    else:
        config_dict["embed"] = embed_config()
    if "optim_loss" in yaml_dict:
        config_dict["optim_loss"] = optim_loss_config(**yaml_dict["optim_loss"])
    else:
        config_dict["optim_loss"] = optim_loss_config()
    if "debug" in yaml_dict:
        config_dict["debug"] = debug_config(**yaml_dict["debug"])
    else:
        config_dict["debug"] = debug_config()

    if (config_dict["data"]["bvp_agg"] is not None) \
            and (config_dict["data"]["bvp_agg"] not in ("stack", "1d", "sum")):
        raise ValueError(f"Parameter bvp_agg is "
                         f"{config_dict['data']['bvp_agg']} but must be one of"
                         f"[`stack`, `1d`, `sum`].")

    return config_dict
