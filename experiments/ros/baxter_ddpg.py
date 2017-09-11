"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.algos.ddpg import DDPG
from railrl.envs.ros.baxter_env import BaxterEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.tensorflow.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic


def example(variant):
    env = BaxterEnv(update_hz=20)
    es = OUStrategy(
        max_sigma=0.05,
        min_sigma=0.05,
        action_space=env.action_space,
    )
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    use_new_version=variant['use_new_version']
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        n_epochs=30,
        batch_size=1024,
        use_new_version=use_new_version,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="6-17-ddpg-baxter-varying-end-effector-norm-distance",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'use_new_version': False,
            }
    )
