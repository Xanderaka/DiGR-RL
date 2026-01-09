"""
Multi Task Head.

Multitask head that predicts both the gesture and the domain.

Author:
    Adapted from original work by Yvan Satyawan and Jonas Niederle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class RevGrad(Function):
    """Gradient Reversal Layer (GRL) for domain adaptation."""
    @staticmethod
    def forward(ctx, input, lambda_=1.0):
        # Save lambda for backward pass
        ctx.lambda_ = lambda_
        # Identity in forward pass
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient during backpropagation
        return -ctx.lambda_ * grad_output, None


class DomainPredictor(nn.Module):
    """Predicts the domain label with a gradient reversal layer for adversarial learning."""
    def __init__(self,
                 in_features: int,
                 num_domains: int,
                 num_layers: int = 2,
                 hidden_dim: int = 64,
                 ac_func: nn.Module = nn.ReLU,
                 lambda_grl: float = 1.0):
        super().__init__()

        self.lambda_grl = lambda_grl

        # Define the fully connected classifier architecture
        layers = []
        dims = [in_features] + [hidden_dim] * (num_layers - 1) + [num_domains]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(ac_func())

        # Final output layer (no activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Apply gradient reversal before domain prediction
        x = RevGrad.apply(x, self.lambda_grl)
        return self.classifier(x)  # raw logits (no softmax)


class GesturePredictor(nn.Module):
    """Fully connected gesture classifier head."""
    def __init__(self,
                 fc_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 num_classes: int = 6,
                 in_features: int = 10,
                 num_layers: int = 3):
        super().__init__()

        def linear_block(in_dim, out_dim):
            """Single linear layer with BN, dropout, and activation."""
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(dropout),
                fc_ac_func()
            )

        # Define layer dimensions exponentially (16, 32, 64, ...)
        layer_dims = [2 ** (i + 4) for i in range(num_layers)]
        self.mlp = nn.Sequential()

        # Build sequential layers: input → hidden layers → output
        layers = [(in_features, layer_dims[0])] + \
                 [(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)] + \
                 [(layer_dims[-1], num_classes)]

        for in_dim, out_dim in layers:
            self.mlp.append(linear_block(in_dim, out_dim))

        # Softmax for class probabilities
        self.mlp.append(nn.Softmax(dim=1))

    def forward(self, x):
        return self.mlp(x)


class MultiTaskHead(nn.Module):
    """Combines gesture and domain prediction heads for multitask learning."""
    def __init__(self,
                 latent_dim: int,
                 domain_label_size: int,
                 num_gesture_classes: int = 6,
                 num_domains: int = 3,
                 gesture_layers: int = 3,
                 domain_layers: int = 2,
                 gesture_dropout: float = 0.3,
                 domain_dropout: float = 0.3,
                 gesture_ac_func: nn.Module = nn.ReLU,
                 domain_ac_func: nn.Module = nn.ReLU):
        super().__init__()

        # Input to heads is latent vector + domain embedding
        input_dim = latent_dim + domain_label_size
        self.domain_label_size = domain_label_size

        # Gesture classification head
        self.gesture_head = GesturePredictor(fc_ac_func=gesture_ac_func,
                                             dropout=gesture_dropout,
                                             num_classes=num_gesture_classes,
                                             in_features=input_dim,
                                             num_layers=gesture_layers)

        # Domain prediction head (with gradient reversal)
        self.domain_head = DomainPredictor(ac_func=domain_ac_func,
                                           num_domains=num_domains,
                                           in_features=input_dim,
                                           num_layers=domain_layers)

    def forward(self, z):
        # Run both prediction heads on shared latent input
        gesture_out = self.gesture_head(z)
        domain_out = self.domain_head(z)
        return gesture_out, domain_out


def run_heads(null_head: MultiTaskHead,
              embed_head: MultiTaskHead,
              null_embedding: torch.Tensor,
              agent_embedding: torch.Tensor,
              z: torch.Tensor):
    """Runs both multitask heads with either null or agent embeddings.

    Args:
        null_head: Head using baseline embedding (no agent influence).
        embed_head: Head using agent-modified embedding (if provided).
        null_embedding: Domain embedding for baseline.
        agent_embedding: Domain embedding modified by RL agent.
        z: Latent vector from encoder.

    Returns:
        gesture_null, domain_null, gesture_embed, domain_embed
    """
    # Concatenate latent features with null embedding
    gesture_null, domain_null = null_head(torch.cat([z, null_embedding], dim=1))

    # Run the RL-modified version if available
    if embed_head is not None:
        gesture_embed, domain_embed = embed_head(torch.cat([z, agent_embedding], dim=1))
    else:
        gesture_embed, domain_embed = None, None

    return domain_null, gesture_null, domain_embed, gesture_embed
