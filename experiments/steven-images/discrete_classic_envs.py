"""
Run DQN on grid world.
"""

import gym
import numpy as np

from rlkit.torch.dqn.double_dqn import DoubleDQN

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import Mlp
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.envs.wrappers import DiscretizeEnv

from rlkit.launchers.launcher_util import setup_logger

def experiment(variant):
    env = gym.make(variant['env_id'])
    training_env = gym.make(variant['env_id'])
    env = DiscretizeEnv(env, variant['bins'])
    training_env = DiscretizeEnv(training_env, variant['bins'])
    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    qf_criterion = variant['qf_criterion_class']()
    algorithm = variant['algo_class'](
        env,
        training_env=training_env,
        qf=qf,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            epsilon=.2,
            tau=0.001,
            hard_update_period=1000,
            replay_buffer_size=10000,
            save_environment=True,  # Can't serialize CartPole for some reason
        ),
        algo_class=DoubleDQN,#DDPG,#DoubleDQN,
        qf_criterion_class=HuberLoss,
        bins=9,
        env_id='InvertedPendulum-v2',
    )
    search_space = {
        'env_id': [
            'Reacher-v2',
        ],
        'bins': [9],
        'algo_class': [
            DoubleDQN,
        ],
        'learning_rate': [
            1e-3,
            1e-4
        ],
        'qf_criterion_class': [
            HuberLoss,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    setup_logger('dqn-images-experiment', variant=variant)
    experiment(variant)

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        #for i in range(2):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="dqn-Pusher2D-test",
                mode='ec2',
                # use_gpu=False,
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
                # use_gpu=True,
            )
