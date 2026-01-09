import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Custom feature extractor: identity mapping
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        self._features_dim = observation_space.shape[0]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations  # Identity


# Custom PPO policy with LeakyReLU and Softmax on actor output
class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.LeakyReLU,
            features_extractor_class=CustomFeatureExtractor
        )
        policy_kwargs.update(kwargs)

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **policy_kwargs
        )

        # Build actor/critic
        self._build(lr_schedule)

        # Add Softmax layer to actor (policy) network output
        if hasattr(self, 'action_net'):
            self.action_net = nn.Sequential(
                self.action_net,     # existing linear layer
                nn.Softmax(dim=-1)   # normalize logits
            )

    def forward(self, obs, deterministic=False):
        return super().forward(obs, deterministic)
