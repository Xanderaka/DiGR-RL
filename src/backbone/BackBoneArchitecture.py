import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from typing import Optional, List

from .DenseNet import DenseNetBackbone
from .ResNet import ResNetBackbone
from .ViT import ViTBackbone
from .CNN import SimpleCNNBackbone

# Wrapper to select and combine different backbone architectures for amplitude and phase signals
class BackBoneArchitecture(nn.Module):
    def __init__(self, ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 latent_dim: int = 10,
                 input_dim: int = 18,
                 initial_kernel_size: int = 3,
                 Backbone_type: str = "",
                 image_size:int = 64,
                 init_channels: int = 64,
                 normalization: bool = False,
                 growth_rate: int = 32,
                 base_channels: int = 128,
                 patch_size: int = 64,
                 vit_depth: int = 6,
                 vit_heads: int = 4,
                 layers_per_stage: Optional[List[int]] = [2, 2, 2, 2]):
        super().__init__()

        # Initialize backbone for amplitude and phase separately
        if Backbone_type == "DenseNet":
            self.amp_backbone = DenseNetBackbone(DNB_ac_func=ac_func, dropout=dropout, latent_dim=latent_dim,
                                                 input_dim=input_dim, initial_kernel_size=initial_kernel_size,
                                                 init_channels=init_channels, use_input_norm=normalization,
                                                 growth_rate=growth_rate)
            self.phase_backbone = DenseNetBackbone(DNB_ac_func=ac_func, dropout=dropout, latent_dim=latent_dim,
                                                   input_dim=input_dim, initial_kernel_size=initial_kernel_size,
                                                   init_channels=init_channels, use_input_norm=normalization,
                                                   growth_rate=growth_rate)
        elif Backbone_type == "ResNet":
            self.amp_backbone = ResNetBackbone(input_dim=input_dim, RNB_ac_func=ac_func,
                                               initial_kernel_size=initial_kernel_size, dropout=dropout,
                                               latent_dim=latent_dim, in_channels=init_channels,
                                               use_input_norm=normalization, layers_per_stage=layers_per_stage,
                                               base_channels=base_channels)
            self.phase_backbone = ResNetBackbone(input_dim=input_dim, RNB_ac_func=ac_func,
                                                 initial_kernel_size=initial_kernel_size, dropout=dropout,
                                                 latent_dim=latent_dim, in_channels=init_channels,
                                                 use_input_norm=normalization, layers_per_stage=layers_per_stage,
                                                 base_channels=base_channels)
        elif Backbone_type == "ViT":
            self.amp_backbone = ViTBackbone(input_dim=input_dim, ViT_ac_func=ac_func, dropout=dropout,
                                            latent_dim=latent_dim, img_size=image_size, base_channels=base_channels,
                                            patch_size=patch_size, vit_depth=vit_depth, vit_heads=vit_heads)
            self.phase_backbone = ViTBackbone(input_dim=input_dim, ViT_ac_func=ac_func, dropout=dropout,
                                              latent_dim=latent_dim, img_size=image_size, base_channels=base_channels,
                                              patch_size=patch_size, vit_depth=vit_depth, vit_heads=vit_heads)
        elif Backbone_type == "CNN":
            self.amp_backbone = SimpleCNNBackbone(input_dim=input_dim, activation_fn=ac_func, dropout=dropout,
                                                  latent_dim=latent_dim, init_channels=init_channels,
                                                  initial_kernel_size=initial_kernel_size)
            self.phase_backbone = SimpleCNNBackbone(input_dim=input_dim, activation_fn=ac_func, dropout=dropout,
                                                    latent_dim=latent_dim, init_channels=init_channels,
                                                    initial_kernel_size=initial_kernel_size)

        self.latent_dim = self.amp_backbone.latent_dim  # Keep track of latent dimension for concatenation

    def forward(self, amp, phase, bvp):
        # Forward amplitude and phase through respective backbones
        z_amp = self.amp_backbone(amp)
        z_phase = self.phase_backbone(phase)

        # Add channel dimension for concatenation
        z_amp = z_amp.unsqueeze(1)
        z_phase = z_phase.unsqueeze(1)

        # Concatenate along feature dimension and flatten to [batch_size, latent_dim*2]
        z = torch.cat((z_amp, z_phase), dim=2).reshape(-1, self.latent_dim * 2)

        return z
