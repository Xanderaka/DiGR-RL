import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic residual block used in ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        # First convolution + batchnorm + ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution + batchnorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection: either identity or projection if size changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)  # Save shortcut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity              # Residual connection
        out = self.relu(out)
        return out

# ResNet backbone
class ResNetBackbone(nn.Module):
    def __init__(self,
                 input_dim=1,
                 in_channels=64,
                 latent_dim=120,
                 layers_per_stage=[2, 2, 2, 2],
                 base_channels=None,
                 initial_kernel_size=7,
                 RNB_ac_func=nn.ReLU,
                 latent_ac_func=nn.Identity,
                 dropout=0.3,
                 use_input_norm=False):
        super(ResNetBackbone, self).__init__()

        self.activation = RNB_ac_func
        self.final_activation = latent_ac_func()
        self.dropout = nn.Dropout(p=dropout)
        self.use_input_norm = use_input_norm
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # Initial convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, in_channels, kernel_size=initial_kernel_size, stride=2,
                      padding=initial_kernel_size // 2, bias=False),
            nn.BatchNorm2d(in_channels),
            self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Default base channels if not provided
        if base_channels is None:
            base_channels = [64, 128, 256, 512]

        assert len(layers_per_stage) <= len(base_channels), \
            "Too many stages for available base_channels."

        # Build residual stages
        self.stages = nn.ModuleList()
        for i, num_blocks in enumerate(layers_per_stage):
            out_channels = base_channels[i]
            stride = 1 if i == 0 else 2  # Downsample after first stage
            stage = self._make_stage(out_channels, num_blocks, stride)
            self.stages.append(stage)

        final_out_channels = base_channels[len(layers_per_stage) - 1]

        # Global average pooling + fully connected projection to latent space
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_latent = nn.Linear(final_out_channels, latent_dim)

    # Helper to build one stage of residual blocks
    def _make_stage(self, out_channels, num_blocks, stride):
        blocks = []
        blocks.append(ResidualBlock(self.in_channels, out_channels, stride=stride))  # First block may downsample
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(self.in_channels, out_channels, stride=1))   # Remaining blocks keep size
        return nn.Sequential(*blocks)

    def forward(self, x):
        # Optional per-image input normalization
        if self.use_input_norm:
            x = (x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True) + 1e-5)

        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        z = self.fc_latent(x)
        z = self.dropout(z)
        z = self.final_activation(z)

        return z


# Example usage
if __name__ == "__main__":
    model = ResNetBackbone(
        input_dim=3,
        in_channels=64,
        latent_dim=120,
        layers_per_stage=[2, 2, 2],
        latent_ac_func=nn.Sigmoid,
        dropout=0.3,
        use_input_norm=True
    )
    x = torch.randn(8, 3, 224, 224)
    features = model(x)
    print(features.shape)  # Output: torch.Size([8, 120])
