import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


def pytorch_init(fan_in: float):
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


def identity(x):
    return x


# Additional Dense layers implementations
class BilinearDense(nn.Module):
    """
    Bilinear implementation as in:
    Multiplicative Interactions and Where to Find Them, ICLR 2020, https://openreview.net/forum?id=rylnK6VtDH
    """
    out_dim: int
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = default_bias_init

    @nn.compact
    def __call__(self, x, z):
        W_layer = nn.Dense(self.out_dim * x.shape[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)
        b_layer = nn.Dense(self.out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)

        W = W_layer(z).reshape(-1, self.out_dim, x.shape[-1])
        b = b_layer(z)[..., None]
        out = (W @ x[..., None]) + b

        return out.squeeze(-1)


class TorchBilinearDense(nn.Module):
    """
    Implementation of the Bilinear layer as in PyTorch:
    https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html#torch.nn.Bilinear
    """
    out_dim: int
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = default_bias_init

    @nn.compact
    def __call__(self, x, z):
        kernel = self.param(
            'kernel', self.kernel_init, (self.out_dim, x.shape[-1], z.shape[-1]), jnp.float32
        )
        bias = self.param('bias', self.bias_init, (self.out_dim, 1), jnp.float32)
        # with same init and inputs this expression gives all True for torch.isclose
        out = ((x.T * (kernel @ z.T)).sum(1) + bias).T
        return out


# Different MLP architectures for RND
class ConcatFirstMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        network = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d + c_d), bias_init=pytorch_init(f_d + c_d)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        ])

        out = network(
            jnp.hstack([feature, context])
        )
        return out


class ConcatLastMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        linear1 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        linear2 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear3 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d + c_d), bias_init=pytorch_init(h_d + c_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        out = nn.relu(linear1(feature))
        out = nn.relu(linear2(out))
        out = nn.relu(linear3(jnp.hstack([out, context])))
        out = linear4(out)
        return out


class ConcatFullMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        linear1 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d + c_d), bias_init=pytorch_init(f_d + c_d))
        linear2 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d + c_d), bias_init=pytorch_init(h_d + c_d))
        linear3 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d + c_d), bias_init=pytorch_init(h_d + c_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        out = nn.relu(linear1(jnp.hstack([feature, context])))
        out = nn.relu(linear2(jnp.hstack([out, context])))
        out = nn.relu(linear3(jnp.hstack([out, context])))
        out = linear4(out)
        return out


class GatedMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        feature_emb = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d)),
            nn.tanh
        ])
        context_emb = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(c_d), bias_init=pytorch_init(c_d)),
            nn.sigmoid
        ])
        combined_emb = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        ])

        out = combined_emb(
            feature_emb(feature) * context_emb(feature)
        )
        return out


class BilinearFirstMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        bilinear = BilinearDense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        combined_emb = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        ])

        out = combined_emb(
            nn.relu(bilinear(feature, context))
        )
        return out


class BilinearLastMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        linear1 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        linear2 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        bilinear3 = BilinearDense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        out = nn.relu(linear1(feature))
        out = nn.relu(linear2(out))
        out = nn.relu(bilinear3(out, context))
        out = linear4(out)
        return out


class BilinearFullMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        bilinear1 = BilinearDense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        bilinear2 = BilinearDense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        bilinear3 = BilinearDense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        out = nn.relu(bilinear1(feature, context))
        out = nn.relu(bilinear2(out, context))
        out = nn.relu(bilinear3(out, context))
        out = linear4(out)
        return out


class TorchBilinearFirstMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        bilinear = TorchBilinearDense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        combined_emb = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        ])

        out = combined_emb(
            nn.relu(bilinear(feature, context))
        )
        return out


class TorchBilinearLastMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        linear1 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        linear2 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        bilinear3 = TorchBilinearDense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        out = nn.relu(linear1(feature))
        out = nn.relu(linear2(out))
        out = nn.relu(bilinear3(out, context))
        out = linear4(out)
        return out


class TorchBilinearFullMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        bilinear1 = TorchBilinearDense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        bilinear2 = TorchBilinearDense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        bilinear3 = TorchBilinearDense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        out = nn.relu(bilinear1(feature, context))
        out = nn.relu(bilinear2(out, context))
        out = nn.relu(bilinear3(out, context))
        out = linear4(out)
        return out


class FilmFirstMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        film = nn.Dense(2 * self.hidden_dim, kernel_init=pytorch_init(c_d), bias_init=pytorch_init(c_d))
        linear1 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        linear2 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear3 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        gamma, beta = jnp.split(film(context), 2, axis=-1)
        out = nn.relu(gamma * linear1(feature) + beta)
        out = nn.relu(linear2(out))
        out = nn.relu(linear3(out))
        out = linear4(out)
        return out


class FilmLastMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        film = nn.Dense(2 * self.hidden_dim, kernel_init=pytorch_init(c_d), bias_init=pytorch_init(c_d))
        linear1 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        linear2 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear3 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        gamma, beta = jnp.split(film(context), 2, axis=-1)
        out = nn.relu(linear1(feature))
        out = nn.relu(linear2(out))
        out = nn.relu(gamma * linear3(out) + beta)
        out = linear4(out)
        return out


class FilmFullMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        film = nn.Dense(2 * self.hidden_dim * 3, kernel_init=pytorch_init(c_d), bias_init=pytorch_init(c_d))
        linear1 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        linear2 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear3 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        gamma, beta = jnp.split(film(context).reshape(context.shape[0], 3, -1), 2, axis=-1)
        out = nn.relu(gamma[:, 0] * linear1(feature) + beta[:, 0])
        out = nn.relu(gamma[:, 1] * linear2(out) + beta[:, 1])
        out = nn.relu(gamma[:, 2] * linear3(out) + beta[:, 2])
        out = linear4(out)
        return out
