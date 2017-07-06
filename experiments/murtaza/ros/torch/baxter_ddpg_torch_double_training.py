"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from os.path import exists
from railrl.envs.ros.baxter_env import BaxterEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
import joblib
def example(variant):
    load_policy_file = variant.get('load_policy_file', None)
    if load_policy_file is not None and exists(load_policy_file):
        data = joblib.load(load_policy_file)
        policy = data['policy']
        qf = data['qf']
        replay_buffer=data['replay_pool']

        use_right_arm = variant['use_right_arm']
        safety_limited_end_effector = variant['safety_limited_end_effector']
        env = BaxterEnv(update_hz=20, use_right_arm=use_right_arm, safety_limited_end_effector=safety_limited_end_effector)
        es = OUStrategy(
            max_sigma=0.05,
            min_sigma=0.05,
            action_space=env.action_space,
        )
        use_new_version = variant['use_new_version']
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
    else:
        use_right_arm = variant['use_right_arm']
        safety_limited_end_effector = variant['safety_limited_end_effector']
        env = BaxterEnv(update_hz=20, use_right_arm=use_right_arm, safety_limited_end_effector=safety_limited_end_effector)
        es = OUStrategy(
            max_sigma=0.05,
            min_sigma=0.05,
            action_space=env.action_space,
        )
        qf = FeedForwardQFunction(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            100,
            100,
        )
        policy = FeedForwardPolicy(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            100,
            100,
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
        exp_prefix="7-5-ddpg-baxter-left-arm-varying-angle-huber-delta-10-TEST-TEST-ETST",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'use_new_version': False,
                'use_right_arm': False,
                'safety_limited_end_effector':False,
                },
        use_gpu=True,
    )
