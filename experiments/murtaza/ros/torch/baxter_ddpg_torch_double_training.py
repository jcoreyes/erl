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
        safety_end_effector_box = variant['safety_end_effector_box']
        env = BaxterEnv(update_hz=20, use_right_arm=use_right_arm, safety_end_effector_box=safety_end_effector_box)
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
        use_right_arm = variant['use_right_arm']
        experiment = variant['experiment']
        loss = variant['loss']
        safety_end_effector_box = variant['safety_end_effector_box']
        remove_action = variant['remove_action']
        magnitude = variant['magnitude']
        temp = variant['temp']
        env = BaxterEnv(experiment=experiment, use_right_arm=use_right_arm, loss=loss, safety_end_effector_box=safety_end_effector_box, remove_action=remove_action, magnitude=magnitude, temp=temp)
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

experiments=['joint_angle|fixed_angle', 'joint_angle|varying_angle', 'end_effector_position|fixed_ee', 'end_effector_position|varying_ee', 'end_effector_position_orientation|fixed_ee', 'end_effector_position_orientation|varying_ee']
if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="7-10-ddpg-baxter-right-arm-fixed-angle-safety-huber-TEST",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'use_target_policy': True,
                'use_right_arm': True,
                'safety_end_effector_box':True,
                'loss':'huber',
                'magnitude':2,
                'temp':1.05,
                'remove_action':False,
                'experiment':experiments[0],
                },
        use_gpu=True,
    )
