import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.policies import BaseModel

# Custom feature extractor that just passes through flattened observations
class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Assumes observations are already flat vectors."""
    def __init__(self, observation_space: gym.Space):
        # features_dim must match output dimension (here, flattened obs)
        super().__init__(observation_space, features_dim=observation_space.shape[0])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations  # Identity mapping


# Custom actor for SAC: outputs mean actions directly for tracing/export
class CustomActor(BaseModel):
    def __init__(self, observation_space, action_space, net_arch):
        super().__init__(observation_space, action_space)
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        # Create MLP for policy latent features
        self.latent_pi = create_mlp(obs_dim, net_arch[-1], net_arch[:-1], activation_fn=nn.LeakyReLU)

        # Final linear layer to output mean action
        self.mu = nn.Linear(net_arch[-1], action_dim)

    def forward(self, obs):
        x = self.latent_pi(obs)
        mean_actions = torch.tanh(self.mu(x))  # Output bounded in [-1,1]
        return mean_actions  # Single tensor output for tracing


# Custom critic network (Q-network) for SAC
class CustomCritic(nn.Module):
    def __init__(self, observation_space, action_space, net_arch):
        super().__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        input_dim = obs_dim + action_dim  # Critic takes concatenated obs + action

        # MLP outputs scalar Q-value
        self.q_network = nn.Sequential(*create_mlp(input_dim, 1, net_arch, activation_fn=nn.LeakyReLU))

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)  # Concatenate along feature dimension
        return self.q_network(x)


# Custom SAC policy using the above actor and critic
class CustomSACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Call parent constructor first
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=CustomFeatureExtractor,  # optional
            **kwargs
        )

        # Explicitly assign custom features extractor, actor, critic, and target critic
        self.features_extractor = CustomFeatureExtractor(observation_space)
        net_arch = [256, 256]
        self.actor = CustomActor(observation_space, action_space, net_arch)
        self.critic = CustomCritic(observation_space, action_space, net_arch)
        self.critic_target = CustomCritic(observation_space, action_space, net_arch)

        # Build parameters for optimizer
        self._build(lr_schedule)


    def _build(self, lr_schedule):
        super()._build(lr_schedule)  # Uses SB3 internal method to initialize optimizers and modules
