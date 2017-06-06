"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG, TargetUpdateMode

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize


def example(*_):
    env = HalfCheetahEnv()
    env = normalize(env)
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
        n_epochs=30,
        batch_size=128,
        # target_update_mode=TargetUpdateMode.HARD,
        # hard_update_period=10000,
        epoch_length=10000,
        eval_samples=1000,
    )
    algorithm.train()


if __name__ == "__main__":
    for seed in range(10):
        run_experiment(
            example,
            exp_prefix="6-5-tf-vs-torch-ddpg-half-cheetah",
            variant=dict(version="TF"),
            seed=seed,
            mode='ec2',
        )
