from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.naf import NAF, NafPolicy

from rlkit.envs.env_utils import gym_env
from rllab.envs.normalized_env import normalize
from rlkit.torch import pytorch_util as ptu
from os.path import exists
import joblib
from rlkit.envs.ros.baxter_env import BaxterEnv
from rlkit.torch import pytorch_util as ptu


def example(variant):
    load_policy_file = variant.get('load_policy_file', None)
    if not load_policy_file == None and exists(load_policy_file):
        data = joblib.load(load_policy_file)
        algorithm = data['algorithm']
        epochs = data['epoch']
        use_gpu = variant['use_gpu']
        if use_gpu and ptu.gpu_enabled():
            algorithm.cuda()
        algorithm.train(start_epoch=epochs)
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
        use_gpu = variant['use_gpu']

        env = BaxterEnv(
            experiment=experiment,
            arm_name=arm_name,
            loss=loss,
            safety_box=safety_box,
            remove_action=remove_action,
            safety_force_magnitude=safety_force_magnitude,
            temp=temp,
            huber_delta=huber_delta,
            reward_magnitude=10,
        )
        es = OUStrategy(
            max_sigma=es_max_sigma,
            min_sigma=es_min_sigma,
            action_space=env.action_space,
        )
        naf_policy = NafPolicy(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            400,
        )
        algorithm = NAF(
            env,
            naf_policy,
            es,
            batch_size=64,
            num_epochs=60,
            num_steps_per_epoch=1000,
            target_hard_update_period=1000,
            max_path_length=100,
            num_steps_per_eval=300,
            naf_policy_learning_rate=1e-3,
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
        exp_prefix="naf-baxter-left-arm-fixed-end-effector",
        seed=0,
        mode='here',
        variant={
            'version': 'Original',
            'arm_name': 'left',
            'safety_box': False,
            'loss': 'huber',
            'huber_delta': .8,
            'safety_force_magnitude': 1,
            'temp': 1.2,
            'remove_action': False,
            'experiment': experiments[2],
            'es_min_sigma': .1,
            'es_max_sigma': .1,
            'num_epochs': 30,
            'batch_size': 1024,
            'use_gpu': True,
        },
        use_gpu=True,
    )
