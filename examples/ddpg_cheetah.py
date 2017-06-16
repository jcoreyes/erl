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


def example(variant):
    env = HalfCheetahEnv()
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
        embedded_hidden_sizes=(300,),
        observation_hidden_sizes=(400,),
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        observation_hidden_sizes=(400, 300),
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            n_epochs=100,
            batch_size=128,
            epoch_length=10000,
            eval_samples=1000,
            max_path_length=1000,
        ),
        version="Example",
    )
    run_experiment(
        example,
        exp_prefix="ddpg-half-cheetah-example",
        variant=variant,
        seed=0,
        mode='here',
    )
