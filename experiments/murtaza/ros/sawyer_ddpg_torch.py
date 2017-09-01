from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.launcher_util import continue_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from os.path import exists
from railrl.envs.ros.sawyer_env import SawyerEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.torch import pytorch_util as ptu
import joblib
import cProfile

def example(variant):
    load_policy_file = variant.get('load_policy_file', None)
    if load_policy_file is not None and exists(load_policy_file):
        data = joblib.load(load_policy_file)
        algorithm = data['algorithm']
        epoch = data['epoch']
        use_gpu = variant['use_gpu']
        if use_gpu and ptu.gpu_enabled():
            algorithm.cuda()
        algorithm.train(start_epoch=epoch)
    else:
        arm_name = variant['arm_name']
        experiment = variant['experiment']
        loss = variant['loss']
        huber_delta = variant['huber_delta']
        safety_force_magnitude = variant['safety_force_magnitude']
        temp = variant['temp']
        es_min_sigma = variant['es_min_sigma']
        es_max_sigma = variant['es_max_sigma']
        num_epochs = variant['num_epochs']
        batch_size = variant['batch_size']
        use_gpu = variant['use_gpu']

        env = SawyerEnv(
            experiment=experiment,
            arm_name=arm_name,
            loss=loss,
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
        if use_gpu and ptu.gpu_enabled():
            algorithm.cuda()
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
        exp_prefix="ddpg-sawyer-fixed-end-effector",
        seed=0,
        mode='here',
        variant={
            'version': 'Original',
            'arm_name': 'right',
            'loss': 'huber',
            'huber_delta': .8,
            'safety_force_magnitude': 3,
            'temp': 5,
            'experiment': experiments[2],
            'es_min_sigma': 1,
            'es_max_sigma': 1,
            'num_epochs': 30,
            'batch_size': 1024,
            'use_gpu':True,
            # 'load_policy_file':'/home/murtaza/Documents/rllab/data/local/08-17-ddpg-sawyer-fixed-angle-safety-check-test/08-17_ddpg-sawyer-fixed-angle-safety-check-test_2017_08_17_11_56_55_0000--s-0/params.pkl'
        },
        use_gpu=True,
    )
    # continue_experiment(
    #     exp_prefix='/home/murtaza/Documents/rllab/data/local/08-17-ddpg-sawyer-fixed-angle-safety-check-test/08-17_ddpg-sawyer-fixed-angle-safety-check-test_2017_08_17_11_56_55_0000--s-0/'
    # )
    # continue_experiment(
    #     exp_dir='/home/murtaza/Documents/rllab/data/local/08-17-ddpg-sawyer-fixed-angle-safety-check-test/08-17_ddpg-sawyer-fixed-angle-safety-check-test_2017_08_17_11_56_55_0000--s-0/'
    #     resume_function=resume_algorithm,
    # )
"""
If i want to continue experiment:
have mod = 'continue'
need to pass in directories too I'm assuming
"""

#set up the argparse for exp directory
