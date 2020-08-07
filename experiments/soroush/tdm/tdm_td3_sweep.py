import argparse
import math

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYEnv
from multiworld.envs.pygame.point2d import Point2DWallEnv

from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_tdm_td3_experiment
import rlkit.misc.hyperparameter as hyp

variant = dict(
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
        td3_kwargs=dict(
        ),
    ),
    qf_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    policy_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    exploration_noise=0.1,
    exploration_type='ou',
    replay_kwargs=dict(
        max_size=int(1E6),
        fraction_goals_are_rollout_goals=0.2,
        fraction_resampled_goals_are_env_goals=0.5,
    ),
    algorithm="TDM-TD3",
    # version="normal",
    env_kwargs=dict(),
    render=False,
    save_video=False,
    dump_video_kwargs=dict(),
	do_state_exp=True,
)

common_params = {
    'exploration_type': ['epsilon'], # ['epsilon', 'ou'], #['epsilon', 'ou', 'gaussian'],
    'algo_kwargs.tdm_kwargs.max_tau': [1, 10, 20, 40, 99], #[10, 20, 50, 99],
}

env_params = {
    'sawyer-reach-xy': { # 6 DoF
        'env_class': [SawyerReachXYEnv],
        'exploration_type': ['epsilon'],
        'env_kwargs.reward_type': ['hand_distance', 'vectorized_hand_distance'],
        'env_kwargs.norm_order': [1, 2],
        'algo_kwargs.base_kwargs.num_epochs': [50],
        'algo_kwargs.tdm_kwargs.max_tau': [1, 10, 25],
        'algo_kwargs.base_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3], #[0.01, 0.1, 1, 10, 100],
        'save_video': [True],
        'save_video_period': [1],
    },
    'sawyer-push-and-reach-xy': {  # 6 DoF
        'env_class': [SawyerPushAndReachXYEnv],
        'env_kwargs.hide_goal_markers': [True],
        'env_kwargs.puck_low': [(-0.2, 0.5)],
        'env_kwargs.puck_high': [(0.2, 0.7)],
        'env_kwargs.hand_low': [(-0.2, 0.5, 0.)],
        'env_kwargs.hand_high': [(0.2, 0.7, 0.5)],
        'env_kwargs.goal_low': [(-0.05, 0.55, 0.02, -0.2, 0.5)],
        'env_kwargs.goal_high': [(0.05, 0.65, 0.02, 0.2, 0.7)],
        'env_kwargs.mocap_low': [(-0.1, 0.5, 0.)],
        'env_kwargs.mocap_high': [(0.1, 0.7, 0.5)],
        'env_kwargs.reward_type': ['vectorized_state_distance'], #['state_distance', 'vectorized_state_distance'],
        'env_kwargs.norm_order': [1], #[1, 2],
        'exploration_type': ['epsilon'], #['epsilon', 'gaussian'],
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [4],
        'algo_kwargs.base_kwargs.num_epochs': [250], #[300],
        'algo_kwargs.tdm_kwargs.max_tau': [20], #[80, 99], #[1, 10, 20, 40, 99],
        'algo_kwargs.base_kwargs.reward_scale': [1e0, 1e2, 1e3], #[1e0, 1e2],
        'algo_kwargs.tdm_kwargs.dense_rewards': [False],
    },
	'point2d-wall': {
		'env_class': [Point2DWallEnv],
        'env_kwargs.wall_shape': ['u'], #['u', ''],
		'env_kwargs.norm_order': [2],
        'env_kwargs.render_onscreen': [False],
        'env_kwargs.grayscale': [False],
		'algo_kwargs.base_kwargs.num_epochs': [30], #[25],
        'algo_kwargs.base_kwargs.reward_scale': [1e1], #[1e0, 1e2], #[1e2],
        'algo_kwargs.tdm_kwargs.max_tau': [10],
		'render_during_eval': [False],
        'save_video': [True],
        'save_video_period': [1],
        'dump_video_kwargs': [
            dict(
                rows=1,
                columns=10,
                pad_length=1,
                pad_color=0,
            ),
        ]
        # 'env_kwargs.fixed_goal': [(0.0, 0.0)],
        # 'env_kwargs.render_dt_msec': [50],
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
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()

    exp_prefix = "tdm-td3-" + args.env
    if len(args.label) > 0:
        exp_prefix = exp_prefix + "-" + args.label

    search_space = common_params
    search_space.update(env_params[args.env])
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    if args.mode == 'ec2' and args.gpu:
        num_exps_per_instance = args.num_seeds
        num_outer_loops = 1
    else:
        num_exps_per_instance = 1
        num_outer_loops = args.num_seeds

    for _ in range(num_outer_loops):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
				grill_tdm_td3_experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=args.gpu,
                num_exps_per_instance=num_exps_per_instance,
                snapshot_gap=int(math.ceil(variant['algo_kwargs']['base_kwargs']['num_epochs'] / 10)),
                snapshot_mode='gap_and_last',
            )