import os
import wandb
import uuid
import pyrallis

import jax
import numpy as np
import jax.numpy as jnp
import optax

from functools import partial
from dataclasses import dataclass, asdict
from flax.core import FrozenDict
from typing import Dict, Tuple, Any, Optional, Callable
from tqdm.auto import trange

from flax.training.train_state import TrainState

from offline_sac.networks import RND, Actor, EnsembleCritic, Alpha
from offline_sac.utils.buffer import ReplayBuffer
from offline_sac.utils.common import Metrics, make_env, evaluate
from offline_sac.utils.running_moments import RunningMeanStd


@dataclass
class Config:
    # wandb params
    project: str = "SAC-RND"
    group: str = "sac-rnd"
    name: str = "sac-rnd"
    # model params
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    beta: float = 1.0
    num_critics: int = 2
    critic_layernorm: bool = True
    # rnd params
    rnd_learning_rate: float = 3e-4
    rnd_hidden_dim: int = 256
    rnd_embedding_dim: int = 32
    rnd_mlp_type: str = "concat_first"
    rnd_target_mlp_type: Optional[str] = None
    rnd_switch_features: bool = True
    rnd_update_epochs: int = 500
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 50
    # general params
    train_seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


class RNDTrainState(TrainState):
    rms: RunningMeanStd


class CriticTrainState(TrainState):
    target_params: FrozenDict


# RND functions
def rnd_bonus(
        rnd: RNDTrainState,
        state: jax.Array,
        action: jax.Array
) -> jax.Array:
    pred, target = rnd.apply_fn(rnd.params, state, action)
    # [batch_size, embedding_dim]
    bonus = jnp.sum((pred - target) ** 2, axis=1) / rnd.rms.std
    return bonus


def update_rnd(
        key: jax.random.PRNGKey,
        rnd: RNDTrainState,
        batch: Dict[str, jax.Array],
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, RNDTrainState, Metrics]:
    def rnd_loss_fn(params):
        pred, target = rnd.apply_fn(params, batch["states"], batch["actions"])
        raw_loss = ((pred - target) ** 2).sum(axis=1)

        new_rms = rnd.rms.update(raw_loss)
        loss = raw_loss.mean(axis=0)
        return loss, new_rms

    (loss, new_rms), grads = jax.value_and_grad(rnd_loss_fn, has_aux=True)(rnd.params)
    new_rnd = rnd.apply_gradients(grads=grads).replace(rms=new_rms)

    # log rnd bonus for random actions
    key, actions_key = jax.random.split(key)
    random_actions = jax.random.uniform(actions_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
    new_metrics = metrics.update({
        "rnd_loss": loss,
        "rnd_rms": new_rnd.rms.std,
        "rnd_data": loss / rnd.rms.std,
        "rnd_random": rnd_bonus(rnd, batch["states"], random_actions).mean()
    })
    return key, new_rnd, new_metrics


def update_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        rnd: RNDTrainState,
        critic: TrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array],
        beta: float,
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, TrainState, jax.Array, Metrics]:
    key, actions_key, random_action_key = jax.random.split(key, 3)

    def actor_loss_fn(params):
        actions_dist = actor.apply_fn(params, batch["states"])
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=actions_key)

        rnd_penalty = rnd_bonus(rnd, batch["states"], actions)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        loss = (alpha.apply_fn(alpha.params) * actions_logp.sum(-1) + beta * rnd_penalty - q_values).mean()

        # logging stuff
        actor_entropy = -actions_logp.sum(-1).mean()
        random_actions = jax.random.uniform(random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
        new_metrics = metrics.update({
            "batch_entropy": actor_entropy,
            "actor_loss": loss,
            "rnd_policy": rnd_penalty.mean(),
            "rnd_random": rnd_bonus(rnd, batch["states"], random_actions).mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean()
        })
        return loss, (actor_entropy, new_metrics)

    grads, (actor_entropy, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return key, new_actor, actor_entropy, new_metrics


def update_alpha(
        alpha: TrainState,
        entropy: jax.Array,
        target_entropy: float,
        metrics: Metrics
) -> Tuple[TrainState, Metrics]:
    def alpha_loss_fn(params):
        alpha_value = alpha.apply_fn(params)
        loss = (alpha_value * (entropy - target_entropy)).mean()

        new_metrics = metrics.update({
            "alpha": alpha_value,
            "alpha_loss": loss
        })
        return loss, new_metrics

    grads, new_metrics = jax.grad(alpha_loss_fn, has_aux=True)(alpha.params)
    new_alpha = alpha.apply_gradients(grads=grads)

    return new_alpha, new_metrics


def update_critic(
        key: jax.random.PRNGKey,
        actor: TrainState,
        rnd: RNDTrainState,
        critic: CriticTrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array],
        gamma: float,
        beta: float,
        tau: float,
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key)

    next_actions_dist = actor.apply_fn(actor.params, batch["next_states"])
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=actions_key)
    rnd_penalty = rnd_bonus(rnd, batch["next_states"], next_actions)

    next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)
    next_q = next_q - alpha.apply_fn(alpha.params) * next_actions_logp.sum(-1) - beta * rnd_penalty

    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    new_critic = new_critic.replace(
        target_params=optax.incremental_update(new_critic.params, new_critic.target_params, tau)
    )
    new_metrics = metrics.update({
        "critic_loss": loss,
        "q_min": q_min,
    })
    return key, new_critic, new_metrics


def update_sac(
        key: jax.random.PRNGKey,
        rnd: RNDTrainState,
        actor: TrainState,
        critic: CriticTrainState,
        alpha: TrainState,
        batch: Dict[str, Any],
        target_entropy: float,
        gamma: float,
        beta: float,
        tau: float,
        metrics: Metrics,
):
    key, new_actor, actor_entropy, new_metrics = update_actor(key, actor, rnd, critic, alpha, batch, beta, metrics)
    new_alpha, new_metrics = update_alpha(alpha, actor_entropy, target_entropy, new_metrics)
    key, new_critic, new_metrics = update_critic(
        key, new_actor, rnd, critic, alpha, batch, gamma, beta, tau, new_metrics
    )
    return key, new_actor, new_critic, new_alpha, new_metrics


def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array) -> jax.Array:
        dist = actor.apply_fn(actor.params, obs)
        action = dist.mean()
        return action
    return _action_fn


@pyrallis.wrap()
def main(config: Config):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    buffer = ReplayBuffer.create_from_d4rl(config.dataset_name, config.normalize_reward)
    state_mean, state_std = buffer.get_moments("states")
    action_mean, action_std = buffer.get_moments("actions")

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, rnd_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)

    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]
    target_entropy = -init_action.shape[-1]

    rnd_module = RND(
        hidden_dim=config.rnd_hidden_dim,
        embedding_dim=config.rnd_embedding_dim,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        mlp_type=config.rnd_mlp_type,
        target_mlp_type=config.rnd_target_mlp_type,
        switch_features=config.rnd_switch_features
    )
    rnd = RNDTrainState.create(
        apply_fn=rnd_module.apply,
        params=rnd_module.init(rnd_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.rnd_learning_rate),
        rms=RunningMeanStd.create()
    )
    actor_module = Actor(action_dim=init_action.shape[-1], hidden_dim=config.hidden_dim)
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )
    alpha_module = Alpha()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=config.alpha_learning_rate)
    )
    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim, num_critics=config.num_critics, layernorm=config.critic_layernorm
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    update_sac_partial = partial(
        update_sac, target_entropy=target_entropy, gamma=config.gamma, beta=config.beta, tau=config.tau
    )

    def rnd_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_rnd, new_metrics = update_rnd(key, carry["rnd"], batch, carry["metrics"])
        carry.update(
            key=key, rnd=new_rnd, metrics=new_metrics
        )
        return carry

    def sac_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_actor, new_critic, new_alpha, new_metrics = update_sac_partial(
            key=key,
            rnd=carry["rnd"],
            actor=carry["actor"],
            critic=carry["critic"],
            alpha=carry["alpha"],
            batch=batch,
            metrics=carry["metrics"]
        )
        carry.update(
            key=key, actor=new_actor, critic=new_critic, alpha=new_alpha, metrics=new_metrics
        )
        return carry

    # metrics
    rnd_metrics_to_log = [
        "rnd_loss", "rnd_rms", "rnd_data", "rnd_random"
    ]
    bc_metrics_to_log = [
        "critic_loss", "q_min", "actor_loss", "batch_entropy",
        "rnd_policy", "rnd_random", "action_mse", "alpha_loss", "alpha"
    ]
    # shared carry for update loops
    update_carry = {
        "key": key,
        "actor": actor,
        "rnd": rnd,
        "critic": critic,
        "alpha": alpha,
        "buffer": buffer,
    }
    # PRETRAIN RND
    # for epoch in trange(config.rnd_update_epochs, desc="RND Epochs"):
    for epoch in range(config.rnd_update_epochs):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(rnd_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=rnd_loop_update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log({"epoch": epoch, **{f"RND/{k}": v for k, v in mean_metrics.items()}})

    # TRAIN BC
    # for epoch in trange(config.num_epochs, desc="SAC Epochs"):
    for epoch in range(config.num_epochs):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=sac_loop_update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log({"epoch": epoch, **{f"SAC/{k}": v for k, v in mean_metrics.items()}})

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            actor_action_fn = action_fn(actor=update_carry["actor"])

            eval_returns = evaluate(eval_env, actor_action_fn, config.eval_episodes, seed=config.eval_seed)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            wandb.log({
                "epoch": epoch,
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })

    wandb.finish()


if __name__ == "__main__":
    main()
