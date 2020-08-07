import random

from rlkit.envs.env_utils import get_dim
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.launchers.launcher_util import resume_torch_algorithm, \
    continue_experiment, run_experiment
from rlkit.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from rlkit.torch.ddpg.ddpg import DDPG
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.torch import pytorch_util as ptu
import sys
from rllab.envs.normalized_env import normalize


def example(variant):
    env_class = variant['env_class']
    env_params = variant['env_params']
    env = env_class(**env_params)
    normalize(env)
    es_class = variant['es_class']
    es_params = dict(
        action_space=env.action_space,
        **variant['es_params']
    )
    use_gpu = variant['use_gpu']
    es = es_class(**es_params)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
        100,
    )
    policy_class = variant['policy_class']
    policy_params = dict(
        obs_dim=get_dim(env.observation_space),
        action_dim=get_dim(env.action_space),
        fc1_size=100,
        fc2_size=100,
    )
    policy = policy_class(**policy_params)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params'],
    )
    if use_gpu and ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


experiments = [
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

max_path_length = 100
if __name__ == "__main__":
    try:
        exp_dir = sys.argv[1]
    except:
        exp_dir = None
    if exp_dir:
        continue_experiment(exp_dir, resume_function=resume_torch_algorithm)
    else:
        run_experiment(
            example,
            exp_prefix="ddpg-cheetah-parallel-TEST",
            seed=random.randint(0, 666),
            mode='local',
            variant={
                'version': 'Original',
                'max_path_length': max_path_length,
                'use_gpu': True,
                'es_class': OUStrategy,
                'env_class': HalfCheetahEnv,
                'policy_class': FeedForwardPolicy,
                'normalize_env': False,
                'env_params': dict(),
                'es_params': {
                    'max_sigma': .25,
                    'min_sigma': .25,
                },
                'algo_params': dict(
                    num_epochs=100,
                    num_steps_per_epoch=10000,
                    num_steps_per_eval=100,
                    use_soft_update=True,
                    tau=1e-2,
                    batch_size=128,
                    max_path_length=1000,
                    discount=0.99,
                    qf_learning_rate=1e-3,
                    policy_learning_rate=1e-4,
                    collection_mode='online',
                ),
            },
            use_gpu=True,
        )
        # variant = {
        #         'version': 'Original',
        #         'max_path_length': max_path_length,
        #         'use_gpu': True,
        #         'es_class': OUStrategy,
        #         'env_class': HalfCheetahEnv,
        #         'policy_class': FeedForwardPolicy,
        #         'normalize_env': False,
        #         'env_params': dict(),
        #         'es_params':{
        #             'max_sigma': .25,
        #             'min_sigma': .25,
        #         },
        #         'algo_params':dict(
        #             num_epochs=100,
        #             num_steps_per_epoch=10000,
        #             num_steps_per_eval=100,
        #             use_soft_update=True,
        #             tau=1e-2,
        #             batch_size=128,
        #             max_path_length=1000,
        #             discount=0.99,
        #             qf_learning_rate=1e-3,
        #             policy_learning_rate=1e-4,
        #             collection_mode='online-parallel',
        #             env_train_ratio=10,
        #         ),
        #     }
        # search_space = {
        #     'algo_params.env_train_ratio':[10, 20, 30, 40, 50]
        # }
        # sweeper = hyp.DeterministicHyperparameterSweeper(
        #     search_space, default_parameters=variant,
        # )
        # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        #     run_experiment(
        #         example,
        #         exp_prefix="ddpg-cheetah-cheetah-relative-ratios",
        #         seed=random.randint(0, 666),
        #         mode='ec2',
        #         variant=variant,
        #         exp_id=exp_id,
        #     )
