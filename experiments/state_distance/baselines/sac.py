import random

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.reacher_7dof import Reacher7DofAngleGoalState, \
    Reacher7DofGoalStateEverything
from railrl.envs.wrappers import normalize_box
from railrl.launchers.launcher_util import run_experiment
from railrl.sac.policies import TanhGaussianPolicy
from railrl.sac.sac import SoftActorCritic
from railrl.torch.networks import FlattenMlp
import railrl.misc.hyperparameter as hyp


def experiment(variant):
    env = variant['env_class']()
    env = normalize_box(env)
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-sac-baseline"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "tdm-half-cheetah-x-vel"

    num_epochs = 100
    num_steps_per_epoch = 50000
    num_steps_per_eval = 50000
    max_path_length = 500

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            batch_size=128,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
        version="SAC-actually-cheetah",
        algorithm="SAC",
    )
    search_space = {
        'env_class': [
            GoalXVelHalfCheetah,
        ],
        'multitask': [False, True],
        'algo_params.reward_scale': [
            1000, 100, 10, 1, 0.1,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                seed=seed,
                variant=variant,
                exp_prefix=exp_prefix,
                mode=mode,
                # exp_prefix="dev-sac-half-cheetah",
                # mode='local',
            )
