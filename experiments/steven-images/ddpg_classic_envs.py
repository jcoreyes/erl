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
from rlkit.envs.mujoco.pusher2d import Pusher2DEnv


def experiment(variant):
    env = Pusher2DEnv()
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
            num_epochs=400,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            batch_size=64,
            max_path_length=100,
            discount=.99,

            use_soft_update=True,
            tau=1e-2,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,

            save_replay_buffer=False,
            replay_buffer_size=int(1E5),
        ),

        algo_class=DDPG,
        qf_criterion_class=nn.MSELoss,
        env_id='Reacher-v2'
    )
    search_space = {
        'env_id': [
            # 'Acrobot-v1',
            #'CartPole-v0',
            'Reacher-v2',
            #'InvertedPendulum-v1',
            # 'CartPole-v1',
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
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
#        for i in range(2):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="DDPG-pusher2D",
                mode='local',
                # use_gpu=False,
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
                # use_gpu=True,
            )
