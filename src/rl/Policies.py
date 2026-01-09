import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.distributions import CategoricalDistribution, DiagGaussianDistribution
from src.rl.custom_ppo_policy import CustomPPOPolicy
from src.rl.custom_sac_policy import CustomSACPolicy


class EBEPBase(nn.Module):
    def setup_dist(self):
        """Setup action distributions for discrete or continuous spaces."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.dist = CategoricalDistribution(self.action_space.n)
        else:
            self.is_discrete = False
            action_dim = self.action_space.shape[0]
            self.dist = DiagGaussianDistribution(action_dim)

            log_std_init = -0.5
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
            self.register_parameter("log_std", self.log_std)

    def _get_action_dist_from_latent(self, latent_pi):
        """
        Get the action distribution using latent_pi directly, no action_net needed.
        Works for both discrete and continuous actions.
        """
        if self.is_discrete:
            scaled_logits = latent_pi / self.tau
            return self.dist.proba_distribution(logits=scaled_logits)
        else:
            scaled_mean = latent_pi / self.tau
            return self.dist.proba_distribution(mean_actions=scaled_mean, log_std=self.log_std)


class EBEPPolicyPPO(CustomPPOPolicy, EBEPBase):
    def __init__(self, observation_space, action_space, lr_schedule, tau=1.0, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.tau = tau
        self.setup_dist()

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        action = distribution.mode() if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        value = self.value_net(latent_vf)
        return action, value, log_prob


class EBEPPolicySAC(CustomSACPolicy, EBEPBase):
    def __init__(self, observation_space, action_space, lr_schedule, tau=1.0, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.tau = tau
        self.setup_dist()

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.actor(features)  # single tensor
        distribution = self._get_action_dist_from_latent(latent_pi)
        action = distribution.mode() if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob




class EDBBase:
    def setup_dist(self):
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.dist = CategoricalDistribution(self.action_space.n)
        else:
            self.is_discrete = False
            action_dim = self.action_space.shape[0]
            self.dist = DiagGaussianDistribution(action_dim)

            log_std_init = -2.0  # Lower log_std = less exploration
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
            self.register_parameter("log_std", self.log_std)

    def _get_action_dist_from_latent(self, latent_pi):
        # Use latent_pi directly instead of action_net
        if self.is_discrete:
            return self.dist.proba_distribution(logits=latent_pi)
        else:
            return self.dist.proba_distribution(mean_actions=latent_pi, log_std=self.log_std)
        
class EDBPolicyPPO(CustomPPOPolicy, EDBBase):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.setup_dist()

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        action = distribution.mode() if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        value = self.value_net(latent_vf)
        return action, value, log_prob

class EDBPolicySAC(CustomSACPolicy, EDBBase):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.setup_dist()

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action = distribution.mode() if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob
    
class AEEPPolicyPPO(CustomPPOPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, lambda_: float = 0.5, tau: float = 1.0, **kwargs):
        self.lambda_ = lambda_
        self.tau = tau
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self._init_ebep_components()
        self._init_edb_components()

    def _init_ebep_components(self):
        if self.is_discrete:
            self.dist_ebep = CategoricalDistribution(self.action_space.n)
        else:
            action_dim = self.action_space.shape[0]
            self.dist_ebep = DiagGaussianDistribution(action_dim)
            log_std_init = -0.5
            self.log_std_ebep = nn.Parameter(torch.ones(action_dim) * log_std_init)
            self.register_parameter("log_std_ebep", self.log_std_ebep)

    def _init_edb_components(self):
        if self.is_discrete:
            self.dist_edb = CategoricalDistribution(self.action_space.n)
        else:
            action_dim = self.action_space.shape[0]
            self.dist_edb = DiagGaussianDistribution(action_dim)
            log_std_init = -2.0
            self.log_std_edb = nn.Parameter(torch.ones(action_dim) * log_std_init)
            self.register_parameter("log_std_edb", self.log_std_edb)

    def _get_mixed_action_dist(self, latent_pi):
        logits = self.action_net(latent_pi)

        if self.is_discrete:
            logits_ebep = logits / self.tau
            dist_ebep = self.dist_ebep.proba_distribution(logits=logits_ebep)
            dist_edb = self.dist_edb.proba_distribution(logits=logits)
            mixed_probs = self.lambda_ * dist_ebep.probs + (1 - self.lambda_) * dist_edb.probs
            mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)
            return torch.distributions.Categorical(probs=mixed_probs)

        else:
            mean_ebep = logits / self.tau
            mean_edb = logits
            mean_mixed = self.lambda_ * mean_ebep + (1 - self.lambda_) * mean_edb
            log_std_mixed = self.lambda_ * self.log_std_ebep + (1 - self.lambda_) * self.log_std_edb
            std_mixed = log_std_mixed.exp()
            return torch.distributions.Independent(torch.distributions.Normal(mean_mixed, std_mixed), 1)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        dist = self._get_mixed_action_dist(latent_pi)
        action = dist.mode() if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(latent_vf)

        return action, value, log_prob
    
class AEEPPolicySAC(CustomSACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, lambda_: float = 0.5, tau: float = 1.0, **kwargs):
        self.lambda_ = lambda_
        self.tau = tau
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self._init_ebep_components()
        self._init_edb_components()

    def _init_ebep_components(self):
        if self.is_discrete:
            self.dist_ebep = CategoricalDistribution(self.action_space.n)
        else:
            action_dim = self.action_space.shape[0]
            self.dist_ebep = DiagGaussianDistribution(action_dim)
            log_std_init = -0.5
            self.log_std_ebep = nn.Parameter(torch.ones(action_dim) * log_std_init)
            self.register_parameter("log_std_ebep", self.log_std_ebep)

    def _init_edb_components(self):
        if self.is_discrete:
            self.dist_edb = CategoricalDistribution(self.action_space.n)
        else:
            action_dim = self.action_space.shape[0]
            self.dist_edb = DiagGaussianDistribution(action_dim)
            log_std_init = -2.0
            self.log_std_edb = nn.Parameter(torch.ones(action_dim) * log_std_init)
            self.register_parameter("log_std_edb", self.log_std_edb)

    def _get_mixed_action_dist(self, latent_pi):
        # Use latent_pi directly instead of action_net
        logits = latent_pi

        if self.is_discrete:
            logits_ebep = logits / self.tau
            dist_ebep = self.dist_ebep.proba_distribution(logits=logits_ebep)
            dist_edb = self.dist_edb.proba_distribution(logits=logits)
            mixed_probs = self.lambda_ * dist_ebep.probs + (1 - self.lambda_) * dist_edb.probs
            mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)
            return torch.distributions.Categorical(probs=mixed_probs)
        else:
            mean_ebep = logits / self.tau
            mean_edb = logits
            mean_mixed = self.lambda_ * mean_ebep + (1 - self.lambda_) * mean_edb
            log_std_mixed = self.lambda_ * self.log_std_ebep + (1 - self.lambda_) * self.log_std_edb
            std_mixed = log_std_mixed.exp()
            return torch.distributions.Independent(torch.distributions.Normal(mean_mixed, std_mixed), 1)

    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)  # call extractor directly
        latent_pi = self.actor(features)          # use actor output
        distribution = self._get_mixed_action_dist(latent_pi)

        action = distribution.mode() if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob