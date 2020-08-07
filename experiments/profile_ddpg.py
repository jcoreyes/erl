"""
Exampling of running DDPG on HalfCheetah.
"""
from rlkit.launchers.launcher_util import run_experiment_here
from rlkit.qfunctions.nn_qfunction import FeedForwardCritic
from rlkit.tf.ddpg import DDPG
from rlkit.tf.policies.nn_policy import FeedForwardPolicy
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy


def example(*_):
    env = HalfCheetahEnv()
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
        n_epochs=25,
        epoch_length=1000,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment_here(
        example,
        exp_prefix="3-5-profile-ddpg-half-cheetah",
        seed=2,
    )
