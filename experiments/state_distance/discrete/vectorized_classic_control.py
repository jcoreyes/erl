import random

import numpy as np

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.cartpole_env import CartPole, CartPoleAngleOnly
from railrl.envs.multitask.mountain_car_env import MountainCar
from railrl.launchers.launcher_util import run_experiment
from railrl.state_distance.tdm_dqn import TdmDqn
from railrl.state_distance.discrete_action_networks import \
    VectorizedDiscreteQFunction, ArgmaxDiscreteTdmPolicy
from railrl.torch.networks import FlattenMlp


def experiment(variant):
    env = variant['env_class']()

    qf = VectorizedDiscreteQFunction(
        observation_dim=int(np.prod(env.observation_space.low.shape)),
        action_dim=env.action_space.n,
        goal_dim=env.goal_dim,
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
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 2
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                batch_size=128,
                max_path_length=200,
                discount=0.99,
            ),
            tdm_kwargs=dict(
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
            goal_dim_weights=[0,0,1,1],
        ),
        env_class=MountainCar,
        # version="fix-max-tau",
        version="sample",
    )
    search_space = {
        'algo_params.dqn_kwargs.use_hard_updates': [True, False],
        'env_class': [
            CartPoleAngleOnly,
            # CartPole,
            # MountainCar,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                # exp_prefix="dev-vectorized-discrete-tdm-classic-control",
                # exp_prefix="vectorized-discrete-tdm-cartpole-weight-angle-only",
                # exp_prefix="vectorized-discrete-tdm-cartpole-no-hack",
                exp_prefix="vectorized-discrete-tdm-cartpole-goal_dim_weights",
                seed=seed,
                variant=variant,
                exp_id=exp_id,
                # mode='ec2',
                # use_gpu=False,
                mode='local',
                use_gpu=True,
            )
