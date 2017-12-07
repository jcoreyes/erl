"""
Run PyTorch DDPG on HalfCheetah.
"""
import numpy as np

from railrl.envs.wrappers import normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from rllab.envs.mujoco.ant_env import AntEnv

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import \
    InvertedDoublePendulumEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from railrl.torch.algos.n3dpg import N3DPG
from railrl.torch.networks import FlattenMlp, MlpPolicy


def example(variant):
    env = variant['env_class']()
    if variant['normalize']:
        env = normalize_box(env)
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
    policy = MlpPolicy(
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
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=101,
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
            hidden_sizes=[100, 100],
        ),
        vf_params=dict(
            hidden_sizes=[100, 100],
        ),
        policy_params=dict(
            hidden_sizes=[100, 100],
        ),
        algorithm="N3DPG",
        version="N3DPG",
        normalize=False,
        env_class=HalfCheetahEnv,
    )
    search_space = {
        'env_class': [
            CartpoleEnv,
            SwimmerEnv,
            HalfCheetahEnv,
            AntEnv,
            HopperEnv,
            InvertedDoublePendulumEnv,
        ],
        # 'algo_params.reward_scale': [
            # 10, 1, 0.1,
        # ],
        'algo_params.tau': [
            1, 1e-2, 1e-3,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(3):
            run_experiment(
                example,
                exp_prefix="n3dpg-many-env-sweep",
                mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=False,
            )
