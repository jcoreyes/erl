from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from os.path import exists
from railrl.envs.ros.sawyer_env_updated import SawyerEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
import joblib

def example(variant):
    #TODO: Fix the loading code to actually work!
    load_policy_file = variant.get('load_policy_file', None)
    if load_policy_file is not None and exists(load_policy_file):
        data = joblib.load(load_policy_file)
        # policy = data['policy']
        # qf = data['qf']
        # replay_buffer = data['replay_pool']
        # env = data['env']
        # es = data['es']
        # epoch = data['epoch']
        algorithm = data['algorithm']
        # batch_size = variant['batch_size']
        # algorithm = DDPG(
        #     env,
        #     qf,
        #     policy,
        #     es,
        #     num_epochs=30-epoch,
        #     batch_size=batch_size,
        #     replay_buffer=replay_buffer,
        # )
        algorithm.policy.cuda()
        algorithm.target_policy.cuda()
        algorithm.qf.cuda()
        algorithm.target_qf.cuda()
        algorithm.train()
    else:
        arm_name = variant['arm_name']
        experiment = variant['experiment']
        loss = variant['loss']
        huber_delta = variant['huber_delta']
        safety_box = variant['safety_box']
        remove_action = variant['remove_action']
        safety_force_magnitude = variant['safety_force_magnitude']
        temp = variant['temp']
        es_min_sigma = variant['es_min_sigma']
        es_max_sigma = variant['es_max_sigma']
        num_epochs = variant['num_epochs']
        batch_size = variant['batch_size']

        env = SawyerEnv(
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
        exp_prefix="7-20-ddpg-sawyer-fixed-angle-huber-move-to-neutral",
        seed=0,
        mode='here',
        variant={
            'version': 'Original',
            'arm_name': 'right',
            'safety_box': True,
            'loss': 'huber',
            'huber_delta': 10,
            'safety_force_magnitude': 2,
            'temp': 1.05,
            'remove_action': False,
            'experiment': experiments[0],
            'es_min_sigma': .1,
            'es_max_sigma': .1,
            'num_epochs': 30,
            'batch_size': 1024,
            'load_policy_file':'/home/murtaza/Documents/rllab/data/local/7-20-ddpg-sawyer-fixed-angle-huber-move-to-neutral/7-20-ddpg-sawyer-fixed-angle-huber-move-to-neutral_2017_07_20_21_45_58_0000--s-0/params.pkl'
        },
        use_gpu=True,
    )
