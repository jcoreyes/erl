import random

from rlkit.envs.wrappers import convert_gym_space
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from rlkit.torch.ddpg import DDPG
from os.path import exists
from rlkit.envs.ros.baxter_env import BaxterEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.torch import pytorch_util as ptu
import joblib
import sys
from rlkit.launchers.launcher_util import continue_experiment
from rlkit.launchers.launcher_util import resume_torch_algorithm
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy

def example(variant):
    env_class = variant['env_class']
    env_params = variant['env_params']
    env = env_class(**env_params)
    obs_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    es_class = variant['es_class']
    es_params = dict(
        action_space=action_space,
        **variant['es_params']
    )
    use_gpu = variant['use_gpu']
    es = es_class(**es_params)
    policy_class = variant['policy_class']
    policy_params = dict(
        obs_dim=int(obs_space.flat_dim),
        action_dim=int(action_space.flat_dim),
        fc1_size=100,
        fc2_size=100,
    )
    policy = policy_class(**policy_params)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
        100,
    )
    algorithm = DDPG(
        env,
        qf,
        policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
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
max_path_length = 1000
left_exp = dict(
    example=example,
    exp_prefix="ddpg-baxter-left-arm-fixed-end-effector-huber-delta-1",
    seed=random.randint(0, 666),
    mode='local',
    variant={
            'version': 'Original',
            'es_class': OUStrategy,
            'env_class': BaxterEnv,
            'policy_class': FeedForwardPolicy,
            'env_params':{
                'arm_name':'left',
                'safety_box':False,
                'loss':'huber',
                'huber_delta':1,
                'experiment':experiments[2],
                'reward_magnitude':1,
            },
            'es_params': {
                'max_sigma': .1,
                'min_sigma': .1,
            },
            'use_gpu':True,
            'algo_params': dict(
                batch_size=1024,
                num_epochs=30,
                num_updates_per_env_step=1,
                num_steps_per_epoch=10000,
                max_path_length=max_path_length,
                num_steps_per_eval=1000,
                ),
            },
    use_gpu=True,
)
right_exp = dict(
    example=example,
    exp_prefix="ddpg-baxter-right-arm-fixed-end-effector-MSE",
    seed=random.randint(0, 666),
    mode='local',
    variant={
        'version': 'Original',
        'es_class': OUStrategy,
        'env_class': BaxterEnv,
        'policy_class': FeedForwardPolicy,
        'env_params': {
            'arm_name': 'right',
            'safety_box': False,
            'loss': 'MSE',
            'huber_delta': 1,
            'experiment': experiments[2],
            'reward_magnitude': 1,
        },
        'es_params': {
            'max_sigma': .1,
            'min_sigma': .1,
        },
        'use_gpu': True,
        'algo_params': dict(
            batch_size=1024,
            num_epochs=30,
            num_updates_per_env_step=1,
            num_steps_per_epoch=10000,
            max_path_length=max_path_length,
            num_steps_per_eval=1000,
        ),
    },
    use_gpu=True,
)

if __name__ == "__main__":
    try:
        exp_dir = sys.argv[1]
    except:
        exp_dir = None

    dictionary = right_exp
    if exp_dir == None:
        run_experiment(
            dictionary['example'],
            exp_prefix=dictionary['exp_prefix'],
            seed=dictionary['seed'],
            mode=dictionary['mode'],
            variant=dictionary['variant'],
            use_gpu=dictionary['use_gpu'],
        )
    else:
        continue_experiment(exp_dir, resume_torch_algorithm)

