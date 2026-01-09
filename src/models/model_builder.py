from time import perf_counter

import torch
import torch.nn as nn

from src.data_utils.widar_dataset import WidarDataset
from src.backbone.BackBoneArchitecture import BackBoneArchitecture
from src.models.multi_task import MultiTaskHead
from src.models.null_agent import NullAgent

# Map string names to PyTorch activation functions
ACTIVATION_FN_MAP = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "selu": nn.SELU,
    "gelu": nn.GELU
}

def compute_num_domains(dataset):
    """Compute the number of unique domains in the dataset."""
    domain_set = set()
    for i in range(len(dataset)):
        _, _, _, info = dataset[i]
        domain_tuple = (
            info['user'],
            info['torso_location'],
            info['face_orientation'],
            info['room_num']
        )
        domain_set.add(domain_tuple)
    return len(domain_set), domain_set

def build_model(config: dict[str, any],
                train_dataset: WidarDataset
                ) -> tuple:
    """
    Builds the DARLInG model: encoder backbone, multitask null head, and null agent.

    Returns:
        backbone: BackBoneArchitecture encoder
        null_head: MultiTaskHead for domain & gesture predictions
        null_agent: NullAgent instance
    """
    print("Building models...")
    start_time = perf_counter()

    # Activation functions
    enc_ac_fn = ACTIVATION_FN_MAP[config["encoder"]["activation_fn"]]
    mt_dec_ac_fn = ACTIVATION_FN_MAP[config["mt"]["d_predictor_activation_fn"]]
    mt_pred_ac_fn = ACTIVATION_FN_MAP[config["mt"]["predictor_activation_fn"]]

    # Sample input to determine dimensions
    x_amp, x_phase, x_bvp, x_info = train_dataset[0]

    # === Encoder Backbone Setup ===
    encoder_input_dim = x_amp.shape[0]
    _, height, width = x_amp.shape
    backbone_type = config["encoder"]["Backbone_type"]
    img_size = max(height, width)

    backbone = BackBoneArchitecture(
        input_dim=encoder_input_dim,
        ac_func=enc_ac_fn,
        initial_kernel_size=config["encoder"]["initial_kernel_size"],
        dropout=config["encoder"]["dropout"],
        latent_dim=config["encoder"]["latent_dim"],
        Backbone_type=backbone_type,
        image_size=img_size,
        normalization=config["encoder"]["normalization"],
        init_channels=config["encoder"]["init_channels"],
        growth_rate=config["encoder"]["growth_rate"],
        base_channels=config["encoder"]["base_channels"],
        patch_size=config["encoder"]["patch_size"],
        vit_depth=config["encoder"]["vit_depth"],
        vit_heads=config["encoder"]["vit_heads"],
        layers_per_stage=config["encoder"]["layers_per_stage"]
    )

    mt_input_head_dim = 2 * config["encoder"]["latent_dim"]

    # === Multitask Null Head ===
    num_domains, _ = compute_num_domains(train_dataset)
    null_head = MultiTaskHead(
        domain_ac_func=mt_dec_ac_fn,
        domain_dropout=config["mt"]["d_predictor_dropout"],
        domain_layers=config["mt"]["d_predictor_num_layers"],
        num_domains=num_domains,
        latent_dim=mt_input_head_dim,
        gesture_layers=config["mt"]["predictor_num_layers"],
        gesture_ac_func=mt_pred_ac_fn,
        gesture_dropout=config["mt"]["predictor_dropout"],
        domain_label_size=len(train_dataset.domain_to_idx)
    )

    # === Null Agent ===
    null_agent = NullAgent(len(train_dataset.domain_to_idx), null_value=0)

    print(f"Completed model building. Took {perf_counter() - start_time:.2f} s.")

    return backbone, null_head, null_agent
