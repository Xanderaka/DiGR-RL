import torch
import torch.nn as nn

# Simple CNN backbone with configurable number of conv blocks and latent dimension
class SimpleCNNBackbone(nn.Module):
    def __init__(self,
                 input_dim: int = 1,
                 latent_dim: int = 128,
                 init_channels: int = 32,
                 initial_kernel_size: int = 3,
                 num_blocks: int = 3,
                 dropout: float = 0.3,
                 activation_fn: nn.Module = nn.ReLU):
        super(SimpleCNNBackbone, self).__init__()

        layers = []
        in_channels = input_dim
        out_channels = init_channels

        for block_idx in range(num_blocks):
            # Use MaxPool for downsampling for all but last block; last block uses AdaptiveAvgPool to 1x1
            if block_idx == num_blocks - 1:
                pooling = nn.AdaptiveAvgPool2d((1, 1))  # Ensures fixed-size output for linear layer
            else:
                pooling = nn.MaxPool2d(2)  # Halve spatial dimensions

            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=initial_kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                activation_fn(),
                pooling
            ])

            # Double the channels for the next block
            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, latent_dim),  # Input is the last number of channels after conv/adaptive pooling
            activation_fn()
        )

        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)  # Output latent vector
        return x


if __name__ == "__main__":
    model = SimpleCNNBackbone(num_blocks=4, activation_fn=nn.LeakyReLU)
    sample_input = torch.randn(1, 1, 224, 224)
    output = model(sample_input)
    print(output.shape)  # torch.Size([1, 128]) by default
