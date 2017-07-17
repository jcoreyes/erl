# import sys
# print(sys.version)
# print(sys.path)
import ipdb; ipdb.set_trace()
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from os.path import exists
from railrl.envs.ros.sawyer_env import SawyerEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
import joblib

def example(variant):
    #TODO: Fix the loading code to actually work!
    load_policy_file = variant.get('load_policy_file', None)
    if load_policy_file is not None and exists(load_policy_file):
        data = joblib.load(load_policy_file)
        policy = data['policy']
        qf = data['qf']
        replay_buffer=data['replay_pool']

        use_right_arm = variant['use_right_arm']
        safety_end_effector_box = variant['safety_end_effector_box']
        env = SawyerEnv(update_hz=20, use_right_arm=use_right_arm, safety_end_effector_box=safety_end_effector_box)
        es = OUStrategy(
            max_sigma=0.05,
            min_sigma=0.05,
            action_space=env.action_space,
        )
        use_target_policy = variant['use_target_policy']
        algorithm = DDPG(
            env,
            es,
            qf=qf,
            policy=policy,
            num_epochs=30,
            batch_size=1024,
            use_target_policy=use_target_policy,
        )
        algorithm.train()
    else:
        experiment = variant['experiment']
        reward_function = variant['reward_function']
        safety_end_effector_box = variant['safety_end_effector_box']
        remove_action = variant['remove_action']
        safety_box_magnitude = variant['safety_box_magnitude']
        safety_box_temp = variant['safety_box_temp']
        env = SawyerEnv(
            experiment=experiment,
            reward_function=reward_function,
            safety_end_effector_box=safety_end_effector_box,
            remove_action=remove_action,
            safety_box_magnitude=safety_box_magnitude,
            safety_box_temp=safety_box_temp
        )
        es = OUStrategy(
            max_sigma=1,
            min_sigma=1,
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
        use_target_policy=variant['use_target_policy']
        algorithm = DDPG(
            env,
            es,
            qf=qf,
            policy=policy,
            num_epochs=30,
            batch_size=1024,
            use_target_policy=use_target_policy,
        )
    algorithm.train()

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]
if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="7-15-ddpg-sawyer-fixed-angle-huber",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'use_target_policy': True,
                'safety_end_effector_box':True,
                'reward_function':'huber',
                'safety_box_magnitude':5,
                'safety_box_temp':1.05,
                'remove_action':True,
                'experiment':experiments[0],
                },
        use_gpu=True,
    )
