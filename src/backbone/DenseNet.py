import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Basic Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, dropout):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Each layer sees all previous feature maps
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate, dropout))
    
    def _make_layer(self, in_channels, growth_rate, dropout):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))  # Concatenate all previous feature maps
            features.append(new_feature)
        return torch.cat(features, dim=1)  # Output all accumulated features

# Transition layer between DenseBlocks to reduce feature maps and downsample
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)  # Reduce spatial resolution
        )
    
    def forward(self, x):
        return self.layer(x)

# DenseNet Backbone for feature extraction
class DenseNetBackbone(nn.Module):
    def __init__(self, 
                 growth_rate=32,
                 num_blocks=3,
                 num_layers_per_block=[6, 12, 24],
                 init_channels=64,
                 DNB_ac_func=nn.ReLU,
                 latent_ac_func=nn.Identity,
                 dropout: float = 0.3,
                 latent_dim: int = 120,
                 input_dim: int = 1,
                 initial_kernel_size: int = 7,
                 use_input_norm: bool = False):
        """
        Args:
            growth_rate: Number of filters added per layer in a DenseBlock.
            num_blocks: Number of DenseBlocks (with transitions in between).
            num_layers_per_block: List specifying number of layers in each block.
            init_channels: Channels after initial convolution.
            latent_dim: Output latent vector size.
            use_input_norm: Optional per-sample input normalization.
        """
        super(DenseNetBackbone, self).__init__()

        self.activation = DNB_ac_func
        self.final_activation = latent_ac_func()
        self.dropout = nn.Dropout(p=dropout)
        self.use_input_norm = use_input_norm
        self.latent_dim = latent_dim

        # Initial convolution + batchnorm + activation + maxpool
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, init_channels,
                      kernel_size=initial_kernel_size, stride=2, padding=initial_kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Downsample early
        )

        # Build DenseBlocks and Transition layers
        self.blocks = nn.ModuleList()
        in_channels = init_channels
        for i in range(num_blocks):
            dense_block = DenseBlock(in_channels, growth_rate, num_layers_per_block[i], dropout)
            self.blocks.append(dense_block)
            in_channels += num_layers_per_block[i] * growth_rate  # Update channel count

            if i != num_blocks - 1:  # Add transition layers except after last block
                trans_layer = TransitionLayer(in_channels, in_channels // 2)
                self.blocks.append(trans_layer)
                in_channels = in_channels // 2

        # Global average pooling to fixed size, then flatten for latent projection
        self.global_pool = nn.AdaptiveAvgPool2d((6, 6))  # Output size 6x6

        self.fc_latent = nn.Linear(in_channels * 6 * 6, latent_dim)

    def forward(self, x):
        if self.use_input_norm:
            # Per-sample normalization (channel-wise mean/std)
            x = (x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True) + 1e-5)

        x = self.conv1(x)

        for block in self.blocks:
            x = block(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        z = self.fc_latent(x)
        z = self.dropout(z)
        z = self.final_activation(z)

        return z

# Example usage
if __name__ == "__main__":
    model = DenseNetBackbone()
    sample_input = torch.randn(1, 1, 224, 224)
    x = model(sample_input)
    print(x.shape)  # Output: torch.Size([1, 120])
