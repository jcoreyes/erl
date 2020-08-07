"""
Test different target update modes.
"""
from rlkit.launchers.algo_launchers import get_env_settings
from rlkit.launchers.launcher_util import run_experiment
from rlkit.qfunctions.nn_qfunction import FeedForwardCritic
from rlkit.tf.ddpg import DDPG, TargetUpdateMode
from rlkit.tf.policies.nn_policy import FeedForwardPolicy
from rllab.exploration_strategies.ou_strategy import OUStrategy


def example(variant):
    env_settings = get_env_settings(
        **variant['env_params']
    )
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **variant['ddpg_params']
    )
    algorithm.train()


if __name__ == "__main__":
    ddpg_params = dict(
        n_epochs=50,
        batch_size=1024,
        epoch_length=10000,
        target_update_mode=TargetUpdateMode.HARD,
        hard_update_period=10000,
    )
    env_params = dict(
        env_id='cheetah',
        normalize_env=False,
    )
    variant = dict(
        ddpg_params=ddpg_params,
        env_params=env_params,
    )
    for seed in range(3):
        run_experiment(
            example,
            exp_prefix="3-3-target-mode-ddpg-cheetah",
            seed=seed,
            variant=variant,
        )
