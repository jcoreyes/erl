import argparse

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from railrl.envs.mujoco.sawyer_push_and_reach_env import SawyerPushAndReachXYEasyEnv
from railrl.envs.mujoco.sawyer_push_and_reach_env import SawyerPushAndReachXYEasyMultiworldEnv

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYEnv
from railrl.envs.mujoco.sawyer_reach_env import SawyerReachXYEnv as SawyerReachXYEnvOld

from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.experiments.soroush.multiworld_tdm import tdm_td3_experiment
import railrl.misc.hyperparameter as hyp

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
        structure='norm_difference',
    ),
    # qf_criterion_class=HuberLoss, # this is not being used in any way, since qf_criterion is not used in td3
    policy_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    exploration_type='ou',
    es_kwargs=dict(),
    replay_buffer_kwargs=dict(
        max_size=int(1E6),
        fraction_goals_are_rollout_goals=0.2,
        fraction_resampled_goals_are_env_goals=0.5,
    ),
    algorithm="TDM-TD3",
    version="normal",
    env_kwargs=dict(
        # fix_goal=False,
        # # fix_goal=True,
        # # fixed_goal=(0, 0.7),
    ),
	reward_params=dict(),
    normalize=False,
    render=False,
    multiworld_env=True,
)

common_params = {
    'exploration_type': ['epsilon', 'ou'], # ['epsilon', 'ou'], #['epsilon', 'ou', 'gaussian'],
    'algo_kwargs.tdm_kwargs.max_tau': [1, 10, 20, 40, 99], #[10, 20, 50, 99],
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
    'sawyer-reach-xy-railrl': {  # 6 DoF
        'env_class': [SawyerReachXYEnvOld],
        'exploration_type': ['epsilon'],
        # 'env_kwargs.reward_type': ['hand_distance'],
        'multiworld_env': [False],
        'algo_kwargs.base_kwargs.num_epochs': [50],
        'algo_kwargs.tdm_kwargs.max_tau': [10, 20, 40],
        'algo_kwargs.base_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3]  # [0.01, 0.1, 1, 10, 100],
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
		# 'reward_params.reward_type': ['state_distance'],
		'reward_params.reward_type': ['state_distance', 'vectorized_state_distance'],
		'reward_params.norm_order': [1, 2], #[1, 2], #[1], [2]
        'qf_kwargs.structure': ['norm_difference', 'none'],
        'exploration_type': ['epsilon'], #['epsilon', 'gaussian'],
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [4], #4
        'algo_kwargs.base_kwargs.num_epochs': [600],
        'algo_kwargs.tdm_kwargs.max_tau': [10, 20, 40], #[1, 10, 20, 40, 99],
        'algo_kwargs.tdm_kwargs.policy_loss_criterion': ['norm', 'mean'], #['norm', 'mean'],
        'algo_kwargs.base_kwargs.reward_scale': [1e0, 1e2, 1e3],  # [0.01, 0.1, 1, 10, 100],
    },
    'sawyer-push-and-reach-xy-railrl': {  # 6 DoF
        'env_class': [SawyerPushAndReachXYEasyEnv],
        'exploration_type': ['epsilon', 'gaussian'],
        'multiworld_env': [False],
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [4],
        'algo_kwargs.base_kwargs.num_epochs': [400], #[1000],
        'algo_kwargs.tdm_kwargs.max_tau': [10, 20, 40],
        'algo_kwargs.base_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3],  # [0.01, 0.1, 1, 10, 100],
    },
    'sawyer-push-and-reach-xy-railrl-for-gm': {  # 6 DoF
        'env_class': [SawyerPushAndReachXYEasyMultiworldEnv],
        'exploration_type': ['epsilon', 'gaussian'],
        'multiworld_env': [True],
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [4],
        'algo_kwargs.base_kwargs.num_epochs': [400],  # [1000],
        'algo_kwargs.tdm_kwargs.max_tau': [10, 20, 40],
        'algo_kwargs.base_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3],  # [0.01, 0.1, 1, 10, 100],
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
                tdm_td3_experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=args.gpu,
                num_exps_per_instance=num_exps_per_instance,
                snapshot_gap=int(variant['algo_kwargs']['base_kwargs']['num_epochs'] / 10),
                snapshot_mode='gap_and_last',
            )