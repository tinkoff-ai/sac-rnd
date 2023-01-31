import wandb
import uuid
import pyrallis

import jax
import numpy as np
import jax.numpy as jnp
import optax

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any, Optional, Callable
from tqdm.auto import trange

from flax.training.train_state import TrainState

from offline_sac.networks import RND, Actor, Alpha
from offline_sac.utils.buffer import ReplayBuffer
from offline_sac.utils.common import Metrics, make_env, evaluate
from offline_sac.utils.running_moments import RunningMeanStd


@dataclass
class Config:
    # wandb params
    project: str = "SAC-RND"
    group: str = "bc-rnd-test"
    name: str = "bc-rnd"
    # model params
    actor_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    hidden_dim: int = 256
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
    num_epochs: int = 100
    num_updates_on_epoch: int = 1000
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
        alpha: TrainState,
        batch: Dict[str, jax.Array],
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, TrainState, jax.Array, Metrics]:
    key, actions_key, random_action_key = jax.random.split(key, 3)

    def actor_loss_fn(params):
        actions_dist = actor.apply_fn(params, batch["states"])
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=actions_key)
        rnd_penalty = rnd_bonus(rnd, batch["states"], actions)

        loss = (alpha.apply_fn(alpha.params) * actions_logp.sum(-1) + rnd_penalty).mean()
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


def update_bc(
        key: jax.random.PRNGKey,
        rnd: RNDTrainState,
        actor: TrainState,
        alpha: TrainState,
        batch: Dict[str, Any],
        metrics: Metrics,
        target_entropy: float
):
    key, new_actor, actor_entropy, new_metrics = update_actor(key, actor, rnd, alpha, batch, metrics)
    new_alpha, new_metrics = update_alpha(alpha, actor_entropy, target_entropy, new_metrics)
    return key, new_actor, new_alpha, new_metrics


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
    buffer = ReplayBuffer.create_from_d4rl(config.dataset_name)
    state_mean, state_std = buffer.get_moments("states")
    action_mean, action_std = buffer.get_moments("actions")

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, rnd_key, actor_key, alpha_key = jax.random.split(key, 4)

    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]

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
    target_entropy = -init_action.shape[-1]

    def rnd_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_rnd, new_metrics = update_rnd(key, carry["rnd"], batch, carry["metrics"])
        carry.update(
            key=key, rnd=new_rnd, metrics=new_metrics
        )
        return carry

    def bc_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_actor, new_alpha, new_metrics = update_bc(
            key, carry["rnd"], carry["actor"], carry["alpha"], batch, carry["metrics"], target_entropy
        )
        carry.update(
            key=key, actor=new_actor, alpha=new_alpha, metrics=new_metrics
        )
        return carry

    # metrics
    rnd_metrics_to_log = [
        "rnd_loss", "rnd_rms", "rnd_data", "rnd_random"
    ]
    bc_metrics_to_log = [
        "actor_loss", "batch_entropy", "rnd_policy", "rnd_random", "action_mse", "alpha_loss", "alpha"
    ]
    # shared carry for update loops
    update_carry = {
        "key": key,
        "rnd": rnd,
        "actor": actor,
        "alpha": alpha,
        "buffer": buffer,
    }
    # PRETRAIN RND
    for epoch in trange(config.rnd_update_epochs, desc="RND Epochs"):
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
    for epoch in trange(config.num_epochs, desc="BC Epochs"):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=bc_loop_update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log({"epoch": epoch, **{f"BC/{k}": v for k, v in mean_metrics.items()}})

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


if __name__ == "__main__":
    main()
