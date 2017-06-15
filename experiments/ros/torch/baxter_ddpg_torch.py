"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG

from railrl.envs.ros.baxter_env import BaxterEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy


def example(variant):
    load_policy_file = variant.get('load_policy_file', None)
    if load_policy_file is not None and exists(load_policy_file):
        with tf.Session():
            data = joblib.load(load_policy_file)
            print(data)
            policy = data['policy']
            qf = data['qf']
            replay_buffer=data['pool']
        env = BaxterEnv(update_hz=20)
        es = OUStrategy(
            max_sigma=0.05,
            min_sigma=0.05,
            env_spec=env.spec,
        )
        use_new_version = variant['use_new_version']
        algorithm = DDPG(
            env,
            es,
            policy,
            qf,
            n_epochs=30,
            batch_size=1024,
            replay_pool=replay_buffer,
            use_new_version=use_new_version,
        )
        algorithm.train()
    else:
        env = BaxterEnv(update_hz=20)
        es = OUStrategy(
            max_sigma=0.05,
            min_sigma=0.05,
            action_space=env.action_space,
        )
        qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
        )
        policy = FeedForwardPolicy(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            400,
            300,
        )
        use_new_version=variant['use_new_version']
        algorithm = DDPG(
            env,
            es,
            qf=qf,
            policy=policy,
            num_epochs=30,
            batch_size=1024,
            use_new_version=use_new_version,
        )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="ddpg-baxter-varying-end-effector-torch",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'use_new_version': False,
            },
        use_gpu=True,
    )
