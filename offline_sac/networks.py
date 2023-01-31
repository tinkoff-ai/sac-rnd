import jax
import distrax
import jax.numpy as jnp
import flax.linen as nn

from typing import Optional

from offline_sac.utils.common import normalize
from offline_sac.nn import (
    GatedMLP,
    ConcatFirstMLP, ConcatLastMLP, ConcatFullMLP,
    BilinearFirstMLP, BilinearLastMLP, BilinearFullMLP,
    TorchBilinearFirstMLP, TorchBilinearLastMLP, TorchBilinearFullMLP,
    FilmFirstMLP, FilmLastMLP, FilmFullMLP
)
from offline_sac.nn import pytorch_init, uniform_init, identity

TYPE_TO_CLS = {
    "gated": GatedMLP,
    "concat_first": ConcatFirstMLP,
    "concat_last": ConcatLastMLP,
    "concat_full": ConcatFullMLP,
    "bilinear_first": BilinearFirstMLP,
    "bilinear_last": BilinearLastMLP,
    "bilinear_full": BilinearFullMLP,
    "torch_bilinear_first": TorchBilinearFirstMLP,
    "torch_bilinear_last": TorchBilinearLastMLP,
    "torch_bilinear_full": TorchBilinearFullMLP,
    "film_full": FilmFullMLP,
    "film_first": FilmFirstMLP,
    "film_last": FilmLastMLP,
}


class RND(nn.Module):
    hidden_dim: int
    embedding_dim: int
    state_mean: jax.Array
    state_std: jax.Array
    action_mean: jax.Array
    action_std: jax.Array
    mlp_type: str = "concat"
    target_mlp_type: Optional[str] = None
    switch_features: bool = False

    def setup(self):
        pred_network_class = TYPE_TO_CLS[self.mlp_type]
        if self.target_mlp_type is None:
            target_network_class = pred_network_class
        else:
            target_network_class = TYPE_TO_CLS[self.target_mlp_type]

        self.predictor = pred_network_class(
            hidden_dim=self.hidden_dim,
            out_dim=self.embedding_dim
        )
        self.target = target_network_class(
            hidden_dim=self.hidden_dim,
            out_dim=self.embedding_dim
        )

    def __call__(self, state, action):
        state = normalize(state, self.state_mean, self.state_std)
        action = normalize(action, self.action_mean, self.action_std)

        if self.switch_features:
            pred = self.predictor(action, state)
            target = self.target(action, state)
        else:
            pred = self.predictor(state, action)
            target = self.target(state, action)

        return pred, jax.lax.stop_gradient(target)


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


# WARN: only for [-1, 1] action bounds, scaling/unscaling is left as an exercise for the reader :D
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state):
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        net = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
        ])
        log_sigma_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))
        mu_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        trunk = net(state)
        mu, log_sigma = mu_net(trunk), log_sigma_net(trunk)
        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = jnp.clip(log_sigma, -5, 2)

        dist = TanhNormal(mu, jnp.exp(log_sigma))
        return dist


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = False

    @nn.compact
    def __call__(self, state, action):
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        network = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d + a_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ])
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = False

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics
        )
        q_values = ensemble(self.hidden_dim, self.layernorm)(state, action)
        return q_values


class Alpha(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param("log_alpha", lambda key: jnp.array([jnp.log(self.init_value)]))
        return jnp.exp(log_alpha)



