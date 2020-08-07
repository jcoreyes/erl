import random

import numpy as np

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.cartpole_env import CartPole, CartPoleAngleOnly
from rlkit.envs.multitask.discrete_reacher_2d import DiscreteReacher2D
from rlkit.envs.multitask.mountain_car_env import MountainCar
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.dqn.policy import ArgmaxDiscretePolicy
from rlkit.state_distance.tdm_dqn import TdmDqn
from rlkit.state_distance.old.discrete_action_networks import \
    VectorizedDiscreteQFunction, ArgmaxDiscreteTdmPolicy
from rlkit.torch.networks import FlattenMlp


def experiment(variant):
    env = variant['env_class']()

    if variant['algo_params']['tdm_kwargs']['vectorized']:
        qf = VectorizedDiscreteQFunction(
            observation_dim=int(np.prod(env.observation_space.low.shape)),
            action_dim=env.action_space.n,
            goal_dim=env.goal_dim,
            **variant['qf_params']
        )
    else:
        qf = FlattenMlp(
            input_size=env.observation_space.low.size + env.goal_dim + 1,
            output_size=env.action_space.n,
            **variant['qf_params']
        )
    policy = ArgmaxDiscreteTdmPolicy(
        qf,
        **variant['policy_params']
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_params']
    )
    algorithm = TdmDqn(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=50,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                num_updates_per_env_step=1,
                batch_size=128,
                max_path_length=200,
                discount=0.99,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
            ),
            dqn_kwargs=dict(
                epsilon=0.2,
                tau=0.001,
                hard_update_period=1000,
            ),
        ),
        her_replay_buffer_params=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            goal_dim_weights=None
        ),
        env_class=MountainCar,
        # version="fix-max-tau",
        version="sample",
    )
    search_space = {
        'algo_params.tdm_kwargs.sample_rollout_goals_from': [
            'fixed',
            # 'environment',
        ],
        'algo_params.tdm_kwargs.cycle_taus_for_rollout': [
            True,
            False,
        ],
        'env_class': [
            DiscreteReacher2D,
            MountainCar,
            CartPole,
            CartPoleAngleOnly,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
                # exp_prefix="dqn-tdm-fixed-goals-various-tasks-check-cycle-tau"
                #            "-short",
                # mode='ec2',
                exp_prefix="dev-dqn-tdm-launch",
                mode='local',
                # use_gpu=True,
            )
