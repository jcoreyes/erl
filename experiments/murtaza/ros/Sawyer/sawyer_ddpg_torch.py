from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.launcher_util import continue_experiment
from railrl.launchers.launcher_util import resume_torch_algorithm
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from railrl.envs.ros.sawyer_env import SawyerEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.torch import pytorch_util as ptu
import sys

def example(variant):
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
    max_path_length = variant['max_path_length']
    reward_magnitude = variant['reward_magnitude']
    safety_box = variant['safety_box']
    env = SawyerEnv(
        experiment=experiment,
        arm_name=arm_name,
        loss=loss,
        safety_force_magnitude=safety_force_magnitude,
        temp=temp,
        huber_delta=huber_delta,
        reward_magnitude=reward_magnitude,
        safety_box=safety_box,
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
        max_path_length=max_path_length,
    )
    if use_gpu and ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()
    env.turn_off_robot()

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

if __name__ == "__main__":
    try:
        exp_dir = sys.argv[1]
    except:
        exp_dir = None

    if exp_dir == None:
        run_experiment(
            example,
            exp_prefix="ddpg-sawyer-fixed-end-effector-lowered-exploration",
            seed=3,
            mode='here',
            variant={
                'version': 'Original',
                'arm_name': 'right',
                'loss': 'huber',
                'huber_delta': 10,
                'safety_force_magnitude': 50,
                'temp': 1,
                'experiment': experiments[2],
                'es_min_sigma': .5,
                'es_max_sigma': .5,
                'num_epochs': 30,
                'batch_size': 1024,
                'use_gpu':True,
                'safety_box':True,
                'max_path_length':100,
                'reward_magnitude':10,
            },
            use_gpu=True,
        )
    else:
        continue_experiment(exp_dir, resume_torch_algorithm)
