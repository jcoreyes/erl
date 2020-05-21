import torch.optim as optim
from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    HopperEnv,
    Walker2dEnv,
    InvertedPendulumEnv, InvertedDoublePendulumEnv)

from railrl.envs.pygame.point2d import Point2DEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from railrl.torch.ddpg.ddpg import DDPG
from railrl.torch.networks import ConcatMlp, TanhMlpPolicy


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    es = GaussianStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'fh-ddpg-vs-ddpg-pendulums-h20'

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=20,
            discount=.99,

            use_soft_update=True,
            tau=1e-2,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,

            save_replay_buffer=False,
            replay_buffer_size=int(1E6),
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        env_kwargs=dict(
        ),
        es_kwargs=dict(
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        ),
        algorithm="DDPG",
        version="DDPG",
    )
    search_space = {
        'env_class': [
            InvertedPendulumEnv,
            InvertedDoublePendulumEnv,
        ],
        'algo_kwargs.num_updates_per_env_step': [1, 5],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
