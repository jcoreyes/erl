import gym

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from torch import nn as nn
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy

def experiment(variant):
    env = gym.make(variant['env_id'])
    env = NormalizedBoxEnv(env)
    es = GaussianStrategy(
        action_space=env.action_space,
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[128, 128]
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[128, 128],
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
        **variant['algo_params']
    )

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=3000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=100,
            discount=.99,

            use_soft_update=True,
            tau=1e-3,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            collection_mode='online-parallel',
            parallel_env_params=dict(num_workers=4),

            save_replay_buffer=False,
            replay_buffer_size=int(1E5),
            num_updates_per_epoch=1,
            reward_scale=.1,
            sim_throttle=True,
        ),

        algo_class=DDPG,
        qf_criterion_class=nn.MSELoss,
    )
    search_space = {
        'env_id': [
            # 'Acrobot-v1',
           # 'CartPole-v0',
            'InvertedDoublePendulum-v2',
           'HalfCheetah-v2',
            'Reacher-v2',
           'InvertedPendulum-v2',
            # 'MountainCar-v0',
        ],
        'algo_class': [
            DDPG,
        ],
        # 'algo_params.use_hard_updates': [True, False],
        'qf_criterion_class': [
            #nn.MSELoss,
            HuberLoss,
        ],
        'algo_params.collection_mode': ['online-parallel']
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(3):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="DDPG-online-parallel-tests-switch-to-multiprocessing-2",
                mode='local',
                use_gpu=False,
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
                # use_gpu=True,
            )
