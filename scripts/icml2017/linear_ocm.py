import random
import sys

from railrl.algos.ddpg_ocm import DdpgOcm
from railrl.qfunctions.memory_qfunction import MemoryQFunction
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.envs.memory.one_char_memory import OneCharMemoryEndOnly
from railrl.policies.linear_ocm_policy import LinearOcmPolicy
from railrl.launchers.launcher_util import setup_logger, set_seed

"""
Set up experiment variants.
"""

num_values = 1
H = 5
batch_size = 64
n_epochs = 100
min_pool_size = 10 * H
replay_pool_size = 1000

n_batches_per_epoch = 100
n_batches_per_eval = 100

n_seeds = 1

for seed in range(n_seeds):
    set_seed(seed)

    epoch_length = H * n_batches_per_epoch
    eval_samples = H * n_batches_per_eval
    max_path_length = H + 1

    ddpg_params = dict(
        batch_size=batch_size,
        n_epochs=n_epochs,
        min_pool_size=min_pool_size,
        replay_pool_size=replay_pool_size,
        epoch_length=epoch_length,
        eval_samples=eval_samples,
        max_path_length=max_path_length,
    )
    variant = dict(
        num_values=num_values,
        H=H,
        ddpg_params=ddpg_params,
        seed=seed,
    )

    """
    Code for running the experiment.
    """

    onehot_dim = num_values + 1

    env = OneCharMemoryEndOnly(n=num_values, num_steps=H)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=onehot_dim,
    )

    policy = LinearOcmPolicy(
        name_or_scope="policy",
        memory_and_action_dim=onehot_dim,
        horizon=H,
        env_spec=env.spec,
    )

    memory_dim = env.memory_dim
    env_action_dim = env.wrapped_env.action_space.flat_dim
    es = NoopStrategy()
    qf = MemoryQFunction(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    algorithm = DdpgOcm(
        env,
        es,
        policy,
        qf,
        **ddpg_params
    )

    setup_logger(
        exp_prefix="2-12-linear_ocm",
        variant=variant,
    )
    algorithm.train()
