from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp
import argparse

from railrl.launchers.exp_launcher import rl_experiment

from multiworld.envs.mujoco.cameras import *

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_ccrig import SawyerMultiobjectEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_leap import SawyerPushAndReachXYEnv

variant = dict(
    rl_variant=dict(
        do_state_exp=True,
        algo_kwargs=dict(
            num_epochs=1000,
            batch_size=128,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
        ),
        max_path_length=100,
        td3_trainer_kwargs=dict(),
        twin_sac_trainer_kwargs=dict(),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
        ),
        exploration_noise=0.1,
        exploration_type='epsilon',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm="TD3",
        dump_video_kwargs=dict(
            rows=1,
            columns=3,
        ),
        save_video_period=50,
    ),
)

x_var = 0.2
x_low = -x_var
x_high = x_var
y_low = 0.5
y_high = 0.7
t = 0.05

env_params = {
    'pnr-leap': {
        'env_class': [SawyerPushAndReachXYEnv],
        'env_kwargs': [dict(
            # hand_low=(-0.20, 0.50),
            # hand_high=(0.20, 0.70),
            # puck_low=(-0.20, 0.50),
            # puck_high=(0.20, 0.70),
            # fix_reset=0.075,
            # sample_realistic_goals=True,
            # reward_type='state_distance',
            # invisible_boundary_wall=True,

            hand_low=(-0.10, 0.50),
            hand_high=(0.10, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            goal_low=(-0.05, 0.55, -0.10, 0.50),
            goal_high=(0.05, 0.65, 0.10, 0.70),
            fix_reset=True,
            fixed_reset=(0.0, 0.4, 0.0, 0.6),
            sample_realistic_goals=False,
            reward_type='hand_and_puck_distance',
            invisible_boundary_wall=True,
        )],
        'init_camera':[sawyer_xyz_reacher_camera_v0],
    },
    'pnr-ccrig': {
        'env_class': [SawyerMultiobjectEnv],
        'env_kwargs': [dict(
            fixed_start=True,
            fixed_colors=False,
            reward_type="dense",
            num_objects=1,
            object_meshes=None,
            num_scene_objects=[1],
            maxlen=0.1,
            action_repeat=1,
            # puck_goal_low=(x_low + 0.01, y_low + 0.01),
            # puck_goal_high=(x_high - 0.01, y_high - 0.01),
            hand_goal_low=(x_low + 3*t, y_low + t),
            hand_goal_high=(x_high - 3*t, y_high -t),
            mocap_low=(x_low + 2*t, y_low , 0.0),
            mocap_high=(x_high - 2*t, y_high, 0.5),
            object_low=(x_low + 0.01, y_low + 0.01, 0.02),
            object_high=(x_high - 0.01, y_high - 0.01, 0.02),
            use_textures=False,

            puck_goal_low=(x_low + 2 * t, y_low),
            puck_goal_high=(x_high - 2 * t, y_high),
        )],
        'init_camera':[sawyer_xyz_reacher_camera_v0],
    },
}

def process_variant(variant):
    rl_variant = variant['rl_variant']

    mpl = rl_variant['max_path_length']
    rl_variant['td3_trainer_kwargs']['discount'] = 1 - 1 / mpl
    rl_variant['twin_sac_trainer_kwargs']['discount'] = 1 - 1 / mpl

    if args.debug:
        rl_variant['algo_kwargs']['num_epochs'] = 4
        rl_variant['algo_kwargs']['batch_size'] = 128
        rl_variant['dump_video_kwargs']['columns'] = 2
        rl_variant['save_video_period'] = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant_only', action='store_true')
    parser.add_argument('--no_video',  action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    search_space = env_params[args.env]
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters(print_info=False)):
        process_variant(variant)
        for _ in range(args.num_seeds):
            run_experiment(
                rl_experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=True,
                time_in_mins=1000,
          )
