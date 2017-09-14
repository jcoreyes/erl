from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.naf import NAF, NafPolicy
from railrl.torch import pytorch_util as ptu
from os.path import exists
import joblib
from railrl.envs.ros.sawyer_env import SawyerEnv
from railrl.torch import pytorch_util as ptu


def example(variant):
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
    naf_policy = NafPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
    )
    algorithm = NAF(
        env,
        naf_policy,
        es,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_path_length=100,
        num_steps_per_epoch=500,
        num_steps_per_eval=500,
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
        exp_prefix="NAF-sawyer-fixed-end-effector-task-test-2",
        seed=0,
        mode='here',
        variant={
            'version': 'Original',
            'arm_name': 'right',
            'safety_box': False,
            'loss': 'huber',
            'huber_delta': 10,
            'safety_force_magnitude': 10,
            'temp': 1,
            'remove_action': False,
            'experiment': experiments[2],
            'es_min_sigma': .25,
            'es_max_sigma': .25,
            'num_epochs': 70,
            'batch_size': 64,
            'use_gpu': True,
        },
        use_gpu=True,
    )
