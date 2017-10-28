import argparse
import joblib
import railrl.torch.pytorch_util as ptu
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.ddpg import DDPG
from railrl.torch.modules import HuberLoss
import random
from pathlib import Path

def experiment(variant):
    path = variant['path']
    data = joblib.load(path)
    env = data['env']
    qf = data['qf']
    policy = data['policy']
    action_space = convert_gym_space(env.action_space)
    es = variant['es_class'](
        action_space=action_space,
        **variant['es_params']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    newpath = Path(path).parent / 'extra_data.pkl'
    extra_data = joblib.load(str(newpath))
    replay_buffer = extra_data.get('replay_buffer', None)
    algo = variant['algo_class'](
        env,
        qf,
        policy,
        exploration_policy,
        replay_buffer=replay_buffer,
        qf_criterion=HuberLoss(),
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train_offline()

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('path', type=str,
                        help='Path to snapshot file to fine tune.')
    args = parser.parse_args()

    max_path_length = 100
    run_experiment(
        experiment,
        exp_prefix="offline-ddpg-final",
        seed=random.randint(0, 666),
        mode='local',
        variant={
            'path': args.path,
            'version': 'Original',
            'max_path_length': max_path_length,
            'use_gpu': True,
            'es_class': OUStrategy,
            'es_params': {
                'max_sigma': .25,
                'min_sigma': .25,
            },
            'algo_class':DDPG,
            'algo_params': dict(
                batch_size=64,
                num_epochs=100,
                max_path_length=max_path_length,
                num_updates_per_env_step=10000,
            ),
        },
        use_gpu=True,
    )
