import numpy as np
import torch.optim as optim
from gym.envs.mujoco import HalfCheetahEnv

from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from railrl.torch.algos.n3dpg import N3DPG
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy


def example(variant):
    env = variant['env_class']()
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    es = OUStrategy(action_space=env.action_space)
    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    qf = FlattenMlp(
        input_size=obs_dim+action_dim,
        output_size=1,
        **variant['vf_params']
    )
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        **variant['vf_params']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_params']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = N3DPG(
        env,
        qf=qf,
        vf=vf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1001,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=64,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            vf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        vf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            hidden_sizes=[300, 300],
        ),
        algorithm="N3DPG",
        version="N3DPG",
        normalize=True,
        env_class=HalfCheetahEnv,
    )
    search_space = {
        'env_class': [
            # CartpoleEnv,
            # SwimmerEnv,
            HalfCheetahEnv,
            # AntEnv,
            # HopperEnv,
            # InvertedDoublePendulumEnv,
        ],
        'algo_kwargs.reward_scale': [
            10000, 100, 1, 0.01,
        ],
        'algo_kwargs.optimizer_class': [
            optim.RMSprop,
            optim.Adam,
            optim.SGD,
        ],
        'algo_kwargs.tau': [
            1e-2,
        ],
        'algo_kwargs.num_updates_per_env_step': [
            1, 5,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(1):
            run_experiment(
                example,
                exp_prefix="n3dpg-cheetah-sweep-huber-2",
                mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=False,
            )
