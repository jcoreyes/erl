from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.envs.wrappers import convert_gym_space
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.algos.parallel_naf import ParallelNAF
from rlkit.torch.naf import NafPolicy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.ros.baxter_env import BaxterEnv
from rlkit.torch import pytorch_util as ptu
import random
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
        hidden_size=100,
        use_batchnorm=False,
    )
    policy = policy_class(**policy_params)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    remote_env = RemoteRolloutEnv(
        env,
        policy,
        exploration_policy,
        variant['max_path_length'],
        variant['normalize_env'],
    )
    algorithm = ParallelNAF(
        remote_env,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params'],
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
import itertools
max_path_length = 100
learning_rates=[1e-3, 5e-3, 1e-4]
use_batch_norm=[True,False]
cart_prod = list(itertools.product(learning_rates, use_batch_norm))
if __name__ == "__main__":
    for _ in range(5):
        for i in range(3):
            run_experiment(
                example,
                exp_prefix="naf-parallel-baxter-fixed-end-effector",
                seed=random.randint(0, 666),
                mode='here',
                variant={
                    'version': 'Original',
                    'max_path_length': max_path_length,
                    'use_gpu': True,
                    'es_class': OUStrategy,
                    'env_class': BaxterEnv,
                    'policy_class': NafPolicy,
                    'normalize_env': False,
                    'env_params': {
                        'arm_name': 'left',
                        'safety_box': False,
                        'loss': 'huber',
                        'huber_delta': 10,
                        'remove_action': False,
                        'experiment': experiments[2],
                        'reward_magnitude': 1,
                    },
                    'es_params': {
                        'max_sigma': .1,
                        'min_sigma': .1,
                    },
                    'algo_params': dict(
                        batch_size=64,
                        num_epochs=30,
                        num_steps_per_epoch=300,
                        target_hard_update_period=10000,
                        max_path_length=max_path_length,
                        num_steps_per_eval=100,
                    ),
                },
                use_gpu=True,
            )
