import argparse
import math

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYEnv

from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_tdm_twin_sac_full_experiment
import rlkit.misc.hyperparameter as hyp

from multiworld.envs.mujoco.cameras import init_sawyer_camera_v4

variant = dict(
    env_kwargs=dict(),
    init_camera=init_sawyer_camera_v4,
    grill_variant=dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=300,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
            ),
            tdm_kwargs=dict(
                max_tau=15,
            ),
            twin_sac_kwargs=dict(
                train_policy_with_reparameterization=True,
                soft_target_tau=1e-3,  # 1e-2
                policy_update_period=1,
                target_update_period=1,  # 1
            ),
        ),
        replay_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        algorithm="GRILL-TDM-TwinSAC",
        # version="normal",
        render=False,
        save_video=False,
        exploration_noise=0.1,
        exploration_type='ou',
        training_mode='train',
        testing_mode='test',
        reward_params=dict(),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
    ),
    train_vae_variant=dict(
        representation_size=16,
        beta=5.0, # should this be 5.0?
        num_epochs=500,
        generate_vae_dataset_kwargs=dict(
            # dataset_path='manual-upload/SawyerPushAndReachXYEnv_1000_init_sawyer_camera_v4_oracleTrue.npy',
            N=1000, # should this be 1000?
            oracle_dataset=True,
            num_channels=3, # should this be 1?
            # use_cached=False,
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-3,
        ),
        vae_kwargs=dict(
            input_channels=3,
        ),
        beta_schedule_kwargs=dict(
            x_values=[0, 100, 200, 500],
            y_values=[0, 0, 5, 5],
        ),
        save_period=5,
    ),
)

common_params = {
    'grill_variant.exploration_type': ['epsilon', 'ou'], # ['epsilon', 'ou'], #['epsilon', 'ou', 'gaussian'],
    'grill_variant.algo_kwargs.tdm_kwargs.max_tau': [1, 10, 20, 40, 99], #[10, 20, 50, 99],
    # 'algo_kwargs.tdm_kwargs.max_tau': [5, 50, 99],
    # 'qf_kwargs.structure': ['none'],
    # 'reward_params.type': [
    #     # 'latent_distance',
    #     # 'log_prob',
    #     # 'mahalanobis_distance'
    # ],
    # 'reward_params.min_variance': [0],
}

env_params = {
    'sawyer-reach-xy': { # 6 DoF
        'env_class': [SawyerReachXYEnv],
        'exploration_type': ['epsilon'],
        'reward_params.reward_type': ['hand_distance', 'vectorized_hand_distance'],
        'reward_params.norm_order': [1, 2],
        'qf_kwargs.structure': ['norm_difference', 'none'],
        'algo_kwargs.base_kwargs.num_epochs': [50],
        'algo_kwargs.tdm_kwargs.max_tau': [1, 10, 25],
        'algo_kwargs.base_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3] #[0.01, 0.1, 1, 10, 100],
    },
    'sawyer-push-and-reach-xy': {  # 6 DoF
        'env_class': [SawyerPushAndReachXYEnv],
        'env_kwargs': [
            dict(
                hide_goal_markers=True,
                puck_low=(-0.2, 0.5),
                puck_high=(0.2, 0.7),
                hand_low=(-0.2, 0.5, 0.),
                hand_high=(0.2, 0.7, 0.5),
                goal_low=(-0.05, 0.55, 0.02, -0.2, 0.5),
                goal_high=(0.05, 0.65, 0.02, 0.2, 0.7),
                mocap_low=(-0.1, 0.5, 0.),
                mocap_high=(0.1, 0.7, 0.5),
            ),
        ],
        'train_vae_variant.num_epochs': [500],
        'train_vae_variant.generate_vae_dataset_kwargs.N': [10000],
        'train_vae_variant.save_period': [20],
        'grill_variant.vae_path': [
            "07-04-grill-tdm-td3-sawyer-push-and-reach-xy-first-attempt/07-04-grill-tdm-td3-sawyer-push-and-reach-xy-first-attempt_2018_07_04_07_18_11_0006--s-20444/vae.pkl"
        ],
        'grill_variant.reward_params.type': ['vectorized_latent_distance'], # ['latent_distance', 'vectorized_latent_distance'],
        'grill_variant.reward_params.norm_order': [1],
        'grill_variant.exploration_type': ['epsilon'],  # ['epsilon', 'gaussian'],
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [4], #[1],
        'grill_variant.algo_kwargs.base_kwargs.num_epochs': [250], #[1000],
        'grill_variant.algo_kwargs.tdm_kwargs.max_tau': [20], #[20, 40],  # [10, 20, 40], #[1, 10, 20, 40, 99],
        'grill_variant.algo_kwargs.base_kwargs.reward_scale': [5e0, 1e1, 2e1, 5e1], #[1e0, 1e1, 1e2, 1e3],  # [1e0, 1e2],
    },
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

    exp_prefix = "grill-tdm-td3-" + args.env
    if len(args.label) > 0:
        exp_prefix = exp_prefix + "-" + args.label

    search_space = common_params
    search_space.update(env_params[args.env])
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    if args.mode == 'ec2' and (not args.no_gpu):
        num_exps_per_instance = args.num_seeds
        num_outer_loops = 1
    else:
        num_exps_per_instance = 1
        num_outer_loops = args.num_seeds

    for _ in range(num_outer_loops):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                grill_tdm_twin_sac_full_experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=(not args.no_gpu),
                num_exps_per_instance=num_exps_per_instance,
                snapshot_gap=int(math.ceil(variant['grill_variant']['algo_kwargs']['base_kwargs']['num_epochs'] / 10)),
                snapshot_mode='gap_and_last',
            )