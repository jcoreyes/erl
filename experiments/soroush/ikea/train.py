from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp
import argparse
import math

from railrl.launchers.exp_launcher import rl_experiment

from furniture.env.furniture_multiworld import FurnitureMultiworld

from multiworld.envs.mujoco.cameras import *

variant = dict(
    rl_variant=dict(
        do_state_exp=True,
        algo_kwargs=dict(
            num_epochs=1000,
            batch_size=2048, #128,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000, #4000,
            min_num_steps_before_training=1000,
        ),
        max_path_length=100,
        td3_trainer_kwargs=dict(
            use_policy_saturation_cost=True,
            policy_saturation_cost_threshold=5.0,
        ),
        twin_sac_trainer_kwargs=dict(),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.5,
            fraction_goals_env_goals=0.0,
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
        use_subgoal_policy=False,
        subgoal_policy_kwargs=dict(
            num_subgoals_per_episode=2,
        ),
        use_masks=False,
        exploration_type='gaussian_and_epsilon',
        es_kwargs=dict(
            max_sigma=.2,
            min_sigma=.2,
            epsilon=.3,
        ),
        algorithm="TD3",
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
        ),
        vis_kwargs=dict(
            vis_list=dict(),
        ),
        save_video_period=50,
    ),
    env_class=FurnitureMultiworld,
    env_kwargs=dict(
        name="FurnitureCursorRLEnv",
        unity=False,
        tight_action_space=False,
        preempt_collisions=True,
        pos_dist=0.1,
        num_connect_steps=0,
        num_connected_ob=True,
        num_connected_reward_scale=5.0,
        goal_type='reset',
        reset_type='var_2dpos', #'var_2dpos+var_1drot', 'var_2dpos+objs_near',
    ),
    imsize=400,
)

env_params = {
    'block-2obj': {
        'env_kwargs.furniture_name': ['block'],

        'env_kwargs.task_type': [
            # 'reach+select+connect+move',
            # 'connect+move',
            # 'reach+select+connect',
            'connect',
        ],
        'env_kwargs.control_degrees': [
            # '3dpos+3drot+select+connect',
            '3dpos+select+connect',
            '2dpos+select+connect',
        ],
        'env_kwargs.reward_type': [
            # 'object_distance',
            # 'object_xyz_distance',
            # 'object1_distance',

            'object1_xyz_distance',
            'object_xyz_in_bounds',
        ],

        'rl_variant.algo_kwargs.num_epochs': [500],
        'rl_variant.save_video_period': [25],  # 50
    },
    'chair-2obj': {
        'env_kwargs.furniture_name': ['chair_agne_0007_2obj'],

        'env_kwargs.task_type': [
            # 'reach+select+connect+move',
            # 'connect+move',
            # 'reach+select+connect',
            'connect',
        ],
        'env_kwargs.control_degrees': [
            '3dpos+3drot+select+connect',
            # '2dpos+select+connect',
        ],
        'env_kwargs.reward_type': [
            # 'object_distance',
            # 'object_xyz_distance',
            # 'object1_distance',

            # 'object1_xyz_distance',
            'object_xyz_in_bounds',
        ],

        'rl_variant.algo_kwargs.num_epochs': [500],
        'rl_variant.save_video_period': [25],  # 50
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
        rl_variant['replay_buffer_kwargs']['max_size'] = int(1e4)
        rl_variant['algo_kwargs']['num_eval_steps_per_epoch'] = 200
        rl_variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 200
        rl_variant['algo_kwargs']['num_trains_per_train_loop'] = 200
        rl_variant['algo_kwargs']['min_num_steps_before_training'] = 200
        rl_variant['dump_video_kwargs']['columns'] = 2
        rl_variant['save_video_period'] = 2
        variant['imsize'] = 250

    if args.no_video:
        rl_variant['save_video'] = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='block-2obj'),
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--max_exps_per_instance', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant_only', action='store_true')
    parser.add_argument('--no_video',  action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'local' and args.label == '':
        args.label = 'local'

    variant['exp_label'] = args.label

    search_space = env_params[args.env]
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    prefix_list = ['train', 'state', args.label]
    while None in prefix_list: prefix_list.remove(None)
    while '' in prefix_list: prefix_list.remove('')
    exp_prefix = '-'.join(prefix_list)

    if args.mode == 'ec2' and (not args.no_gpu):
        max_exps_per_instance = args.max_exps_per_instance
    else:
        max_exps_per_instance = 1

    num_exps_for_instances = np.ones(int(math.ceil(args.num_seeds / max_exps_per_instance)), dtype=np.int32) \
                             * max_exps_per_instance
    num_exps_for_instances[-1] -= (np.sum(num_exps_for_instances) - args.num_seeds)

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters(print_info=False)):
        process_variant(variant)
        for num_exps in num_exps_for_instances:
            run_experiment(
                rl_experiment,
                exp_folder=args.env,
                exp_prefix=exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=(not args.no_gpu),
                gpu_id=args.gpu_id,

                num_exps_per_instance=int(num_exps),

                snapshot_gap=50,
                snapshot_mode="none", #'gap_and_last',
          )
