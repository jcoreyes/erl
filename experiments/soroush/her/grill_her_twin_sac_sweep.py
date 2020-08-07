import argparse
import math

import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1, \
    init_sawyer_camera_v2, init_sawyer_camera_v3
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import (
    SawyerReachXYEnv, SawyerReachXYZEnv
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.experiments.soroush.multiworld_grill import grill_her_twin_sac_full_experiment

variant = dict(
    env_kwargs=dict(
        hide_goal_markers=True,
        puck_low=(-0.05, 0.6),
        puck_high=(0.05, 0.7),
    ),
    init_camera=init_sawyer_camera_v3,
    grill_variant=dict(
        algo_kwargs=dict(
            num_epochs=500,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            # num_epochs=50,
            # num_steps_per_epoch=100,
            # num_steps_per_eval=100,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            num_updates_per_env_step=4, #4
            soft_target_tau=1e-3,
            target_update_period=1,
            policy_update_period=1,
            policy_update_minq=True,
            train_policy_with_reparameterization=True,
        ),
        replay_kwargs=dict(
            max_size=int(1e6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        algorithm='GRILL-HER-Twin-SAC',
        normalize=False,
        render=False,
        save_video=False,
        training_mode='train',
        testing_mode='test',
        reward_params=dict(
            type='latent_distance',
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
    ),
    train_vae_variant=dict(
        representation_size=16,
        beta=5.0,
        num_epochs=500,
        generate_vae_dataset_kwargs=dict(
            N=1000,
            oracle_dataset=True,
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-3,
        ),
        beta_schedule_kwargs=dict(
            x_values=[0, 100, 200, 500],
            y_values=[0, 0, 5, 5],
        ),
        save_period=5,
    ),
)

common_params = {
    # 'normalize': [False, True],
}

env_params = {
    'sawyer-reach-xy': { # 6 DoF
        'env_class': [SawyerReachXYEnv],
        # 'env_kwargs.reward_type': ['hand_distance'],
        'grill_variant.algo_kwargs.num_epochs': [50],
        'grill_variant.algo_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3] #[0.01, 0.1, 1, 10, 100],
    },
    'sawyer-push-and-reach-xy': {  # 6 DoF
        'env_class': [SawyerPushAndReachXYEnv],
        # 'env_kwargs.reward_type': ['puck_distance'],
        'grill_variant.algo_kwargs.discount': [0.98],
        'grill_variant.algo_kwargs.num_epochs': [1000],
        'grill_variant.algo_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3],  # [0.01, 0.1, 1, 10, 100],
    },
}

search_space = {
    # 'grill_variant.training_mode': ['test'],
    # 'grill_variant.observation_key': ['latent_observation'],
    # 'grill_variant.desired_goal_key': ['state_desired_goal'],
    # 'grill_variant.observation_key': ['state_observation'],
    # 'grill_variant.desired_goal_key': ['latent_desired_goal'],
    # 'grill_variant.vae_paths': [
    #     {"16": "/home/vitchyr/git/rlkit/data/doodads3/06-12-dev/06-12"
    #            "-dev_2018_06_12_18_57_14_0000--s-28051/vae.pkl",
    #      }
    # ],
    # 'grill_variant.rdim': ["16"],
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='sawyer-reach-xy')
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument("--no_gpu", action="store_true")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()

    exp_prefix = "grill-twin-sac-" + args.env
    if len(args.label) > 0:
        exp_prefix = exp_prefix + "-" + args.label

    search_space = common_params
    search_space.update(env_params[args.env])
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(args.num_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                grill_her_twin_sac_full_experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=(not args.no_gpu),
                snapshot_gap=int(math.ceil(variant['grill_variant']['algo_kwargs']['num_epochs'] / 10)),
                snapshot_mode='gap_and_last',
            )
