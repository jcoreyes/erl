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
        # policy = data['policy']
        # qf = data['qf']
        # replay_buffer=data['replay_pool']
        # env = data['env']
        # es = data['es']
        # epoch = data['epoch']
        algorithm = data['algorithm']
        # use_target_policy = variant['use_target_policy']
        #
        # algorithm = DDPG(
        #     env,
        #     es,
        #     qf=qf,
        #     policy=policy,
        #     num_epochs=30-epoch,
        #     batch_size=1024,
        #     use_target_policy=use_target_policy,
        #     replay_buffer=replay_buffer,
        # )
        algorithm.train()
    else:
        arm_name = variant['arm_name']
        experiment = variant['experiment']
        loss = variant['loss']
        huber_delta= variant['huber_delta']
        safety_box = variant['safety_box']
        remove_action = variant['remove_action']
        safety_force_magnitude = variant['safety_force_magnitude']
        temp = variant['temp']
        es_min_sigma = variant['es_min_sigma']
        es_max_sigma = variant['es_max_sigma']
        num_epochs = variant['num_epochs']
        batch_size = variant['batch_size']
        
        env = BaxterEnv(
            experiment=experiment,
            arm_name=arm_name,
            loss=loss,
            safety_box=safety_box,
            remove_action=remove_action,
            safety_force_magnitude=safety_force_magnitude,
            temp=temp,
            huber_delta=huber_delta,
        )
        es = OUStrategy(
            max_sigma=es_max_sigma,
            min_sigma=es_min_sigma,
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
        algorithm = DDPG(
            env,
            qf,
            policy,
            es,
            num_epochs=num_epochs,
            batch_size=batch_size,
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
        exp_prefix="7-19-ddpg-baxter-right-arm-fixed-angle-huber-safety-REFACTOR-TEST",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'arm_name':'right',
                'safety_box':True,
                'loss':'huber',
                'huber_delta':10,
                'safety_force_magnitude':1,
                'temp':1.05,
                'remove_action':False,
                'experiment':experiments[0],
                'es_min_sigma':.05,
                'es_max_sigma':.05,
                'num_epochs':30,
                'batch_size':1024,
                },
        use_gpu=True,
    )
