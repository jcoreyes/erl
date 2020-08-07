import argparse
import math
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYEnv, SawyerReachXYZEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_twin_sac_experiment
import rlkit.misc.hyperparameter as hyp

variant = dict(
    algo_kwargs=dict(
        base_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=100,
            num_updates_per_env_step=1,
            batch_size=128,
            discount=0.99,
            reward_scale=1,
        ),
        her_kwargs=dict(
        ),
        twin_sac_kwargs=dict(
            train_policy_with_reparameterization=True,
            soft_target_tau=1e-3,  # 1e-2
            policy_update_period=1,
            target_update_period=1,  # 1
            use_automatic_entropy_tuning=True,
        ),
    ),
    qf_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    vf_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    policy_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    exploration_noise=0,
    exploration_type='ou',
    replay_buffer_kwargs=dict(
        max_size=int(1E6),
        fraction_goals_are_rollout_goals=0,
        fraction_resampled_goals_are_env_goals=0.5,
    ),
    algorithm="HER-Twin-SAC",
    version="normal",
    render=False,
    save_video=True,
    save_video_period=100,
    imsize=48,
    init_camera=sawyer_pusher_camera_upright_v2,
    do_state_exp=True,
)

common_params = {
}

env_params = {
    'door': {
        'env_id':['SawyerDoorHookResetFreeEnv-v1'],
        'algo_kwargs.twin_sac_kwargs.train_policy_with_reparameterization': [True],
        'save_video':[False]
    },
    'small_pusher': {
        'env_class':[SawyerPushAndReachXYEnv],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals':[1/2],
        'algo_kwargs.base_kwargs.max_path_length':[500],
        'algo_kwargs.base_kwargs.num_steps_per_eval':[5000],
        'env_kwargs':[dict(
        hand_low=(-0.16, 0.4, 0.05),
        hand_high=(0.16, 0.75, 0.3),
        puck_low=(-.4, .2),
        puck_high=(.4, 1),
        goal_low=(-0.15, 0.4, 0.02, -.1, .5),
        goal_high=(0.15, 0.74, 0.02, .1, .7),
        xml_path='sawyer_xyz/sawyer_push_puck_smaller_arena.xml',
        num_resets_before_puck_reset=int(1e6),
        num_resets_before_hand_reset=int(1e6),
        reward_type='puck_distance',
    )],
    },
    'pusher':{
    'env_class':[SawyerPushAndReachXYEnv],
    'algo_kwargs.base_kwargs.max_path_length':[500],
    'algo_kwargs.base_kwargs.num_steps_per_eval':[5000],
    'replay_buffer_kwargs.fraction_goals_are_rollout_goals':[1/2],
    'env_kwargs':[dict(
        reward_type='puck_distance',
        hand_low=(-0.28, 0.3, 0.05),
        hand_high=(0.28, 0.9, 0.3),
        puck_low=(-.4, .2),
        puck_high=(.4, 1),
        goal_low=(-0.25, 0.3, 0.02, -.2, .4),
        goal_high=(0.25, 0.875, 0.02, .2, .8),
        num_resets_before_puck_reset=int(1e6),
        num_resets_before_hand_reset=int(1e6),
    )],
    },
    'reacher': {
        'env_class': [SawyerReachXYEnv, SawyerReachXYZEnv],
        'algo_kwargs.twin_sac_kwargs.train_policy_with_reparameterization': [True, False],
    }
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='door')
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()

    exp_prefix = "her-twin-sac-" + args.env
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
                grill_her_twin_sac_experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=args.gpu,
                num_exps_per_instance=num_exps_per_instance,
                snapshot_gap=int(math.ceil(variant['algo_kwargs']['base_kwargs']['num_epochs'] / 10)),
                snapshot_mode='gap_and_last',
            )
