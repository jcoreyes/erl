"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.torch.ddpg import DDPG

from railrl.envs.ros.baxter_env import BaxterEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy


def example(variant):
    oad_policy_file = variant.get('load_policy_file', None)
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
            action_space=env.action_space,
        )
        use_new_version = variant['use_new_version']
        algorithm = DDPG(
            env,
            es,
            policy,
            qf,
            n_epochs=2,
            batch_size=1024,
            replay_pool=replay_buffer,
            use_new_version=use_new_version,
        )
        algorithm.train()

    env = BaxterEnv(update_hz=20)
    es = OUStrategy(
        max_sigma=0.05,
        min_sigma=0.05,
        env_spec=env.spec,
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
        num_epochs=30,
        batch_size=1024,
        use_new_version=use_new_version,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="ddpg-baxter-fixed-angle-torch",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'use_new_version': False,
            }
    )
