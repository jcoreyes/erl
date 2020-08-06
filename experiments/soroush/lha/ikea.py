import railrl.misc.hyperparameter as hyp
from railrl.misc.exp_util import (
    run_experiment,
    parse_args,
    preprocess_args,
)
from railrl.launchers.exp_launcher import rl_experiment
from furniture.env.furniture_multiworld import FurnitureMultiworld

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
        sac_trainer_kwargs=dict(
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale=100,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
            recompute_rewards=False,
            ob_keys_to_save=[
                'oracle_connector_info',
                'oracle_robot_info',
            ],
        ),
        contextual_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_future_context=0.0,
            fraction_distribution_context=0.0,
            recompute_rewards=False,
            observation_keys=[
                'oracle_connector_info',
                'oracle_robot_info',
            ],
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
        exploration_noise=0.6,
        # es_kwargs=dict(
        #     max_sigma=.2,
        #     min_sigma=.2,
        #     epsilon=.3,
        # ),
        algorithm="td3",
        context_based=True,
        save_video=True,
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
            pad_color=0,
            pad_length=0,
            subpad_length=0,
        ),
        vis_kwargs=dict(
            vis_list=dict(),
        ),
        save_video_period=50,
        renderer_kwargs=dict(),
        task_variant=dict(
            task_conditioned=False,
        ),
        mask_variant=dict(
            mask_conditioned=False,
        ),
    ),
    env_class=FurnitureMultiworld,
    env_kwargs=dict(
        name="FurnitureCursorRLEnv",
        unity=False,
        tight_action_space=True,
        preempt_collisions=True,
        boundary=[0.5, 0.5, 1.2],
        pos_dist=0.2,
        num_connect_steps=0,
        num_connected_ob=True,
        num_connected_reward_scale=5.0,
        goal_type='zeros', #reset
        reset_type='var_2dpos+no_rot', #'var_2dpos+var_1drot', 'var_2dpos+objs_near',

        task_type='connect',
        control_degrees='3dpos+select+connect',
        obj_joint_type='slide',
        connector_ob_type='dist',

        clip_action_on_collision=True,

        light_logging=True,
    ),
    imsize=400,
)

env_params = {
    'block-2obj': {
        'env_kwargs.furniture_name': ['block'],
        'env_kwargs.reward_type': [
            'oib+nc+conn_dist',
        ],

        'rl_variant.algo_kwargs.num_epochs': [500],
        'rl_variant.save_video_period': [25],  # 50
    },
    'chair-2obj': {
        'env_kwargs.furniture_name': ['chair_agne_0007_2obj'],
        'env_kwargs.reward_type': [
            'oib+nc',
            'oib+nc+sel_conn_dist',
        ],

        'rl_variant.algo_kwargs.num_epochs': [500],
        'rl_variant.save_video_period': [25],  # 50
    },
    'chair': {
        'env_kwargs.furniture_name': ['chair_agne_0007'],
        'env_kwargs.reward_type': [
            'oib+nc+first_conn_dist',
            'oib+nc+first_conn_dist+cursor_dist',
            'oib+nc+first_conn_dist+cursor_dist+cursor_sparse_dist',
            'oib+nc+next_conn_dist',
            'oib+nc+next_conn_dist+cursor_dist',
            'oib+nc+next_conn_dist+cursor_dist+cursor_sparse_dist',
            'oib+nc+sel_conn_dist',
            'oib+nc+sel_conn_dist+cursor_dist',
            'oib+nc+sel_conn_dist+cursor_dist+cursor_sparse_dist',

        ],
        'env_kwargs.task_connect_sequence': [
            [0, 1, 2], # leg1 -> leg2 -> seat
        ],

        'rl_variant.algo_kwargs.num_trains_per_train_loop': [1000],
        'rl_variant.algo_kwargs.num_epochs': [1500],

        'rl_variant.save_video_period': [50],  # 25
    },
    'shelf': {
        'env_kwargs.furniture_name': ['shelf_ivar_0678_bb'],
        'env_kwargs.anchor_objects': [['1_column']],
        'env_kwargs.reward_type': [
            # 'nc',
            # 'nc+next_conn_dist',
            # 'nc+next_conn_dist+cursor_dist',
            'nc+next_conn_dist+cursor_dist+cursor_sparse_dist',
            # 'nc+cursor_dist+cursor_sparse_dist',
            # 'cursor_dist',
            # 'cursor_dist+cursor_sparse_dist',
        ],

        'rl_variant.max_path_length': [75],

        'env_kwargs.task_connect_sequence': [[0, 1, 2, 3, 4, 5]],  # col -> box1 -> box2 -> box3 -> box4 -> box5
        'rl_variant.task_variant.task_conditioned': [
            True,
            # False,
        ],
        'rl_variant.task_variant.task_ids': [
            # [1],
            [3],
            # [1, 2, 3],
            # [1, 2, 3, 4, 5],
        ],

        'rl_variant.task_variant.rotate_task_freq_for_expl': [0.25],
        'rl_variant.task_variant.rotate_task_freq_for_eval': [1.0],

        'env_kwargs.task_type': [
            # "connect",
            # "select2+connect",

            "reach2+select2+connect",
            # "reach2+select2",
        ],

        'rl_variant.td3_trainer_kwargs.reward_scale': [1000],

        # 'env_kwargs.select_next_obj_only': [True],

        'rl_variant.algo_kwargs.num_epochs': [500],
        'rl_variant.save_video_period': [50],  # 25
    },
    'shelf-4obj': {
        'env_kwargs.furniture_name': ['shelf_ivar_0678_4obj_bb'],
        'env_kwargs.anchor_objects': [['1_column']],
        'env_kwargs.reward_type': [
            # 'nc',
            # 'nc+next_conn_dist',
            # 'nc+next_conn_dist+cursor_dist',
            # 'nc+next_conn_dist+cursor_dist+cursor_sparse_dist',
            # 'nc+cursor_dist+cursor_sparse_dist',
            # 'cursor_dist',
            # 'cursor_dist+cursor_sparse_dist',

            # 'next_conn_dist',
            # 'nc+next_conn_dist',
            'next_conn_dist+cursor_dist',
            # 'nc+next_conn_dist+cursor_dist',
            # 'next_conn_dist+cursor_dist+cursor_sparse_dist',
            # 'nc+next_conn_dist+cursor_dist+cursor_sparse_dist',
        ],

        'rl_variant.max_path_length': [75],

        'env_kwargs.task_connect_sequence': [
            # [0, 1, 2, 3],  # col -> box1 -> box2 -> box3
            [0, 3, 2, 1],
        ],

        'rl_variant.task_variant.task_conditioned': [True],
        'rl_variant.task_variant.task_ids': [
            # [1],
            [3],
            # [1, 2, 3],
        ],
        'rl_variant.task_variant.rotate_task_freq_for_expl': [0.25],
        'rl_variant.task_variant.rotate_task_freq_for_eval': [1.0],

        # 'rl_variant.mask_variant.mask_conditioned': [True],
        # 'rl_variant.mask_variant.mask_idxs': [
        #     # [[3, 4, 5], [17, 18, 19]],
        #     [[3, 4, 5]],
        #     [[17, 18, 19]],
        # ],
        # 'rl_variant.contextual_replay_buffer_kwargs.fraction_future_context': [0.4],
        # 'rl_variant.contextual_replay_buffer_kwargs.fraction_distribution_context': [0.4],
        # 'rl_variant.contextual_replay_buffer_kwargs.recompute_rewards': [True],
        # 'env_kwargs.goal_type': ['assembled'],

        'env_kwargs.task_type': [
            # "move2",
            "select2+move2",
            "reach2+select2+move2",
        ],

        'rl_variant.td3_trainer_kwargs.reward_scale': [1000],
        'rl_variant.algorithm': [
            # 'td3',
            'sac',
        ],


        # 'env_kwargs.select_next_obj_only': [True],

        'rl_variant.algo_kwargs.num_epochs': [500], #500
        'rl_variant.save_video_period': [50],  # 25
    },
    'swivel': {
        'env_kwargs.furniture_name': ['swivel_chair_0700'],
        'env_kwargs.pos_dist': [
            0.2,
        ],

        'env_kwargs.reward_type': [
            'oib+nc+conn_dist',
        ],

        'rl_variant.algo_kwargs.num_epochs': [500],
        'rl_variant.save_video_period': [25],  # 50
    },
}

def process_variant(variant):
    rl_variant = variant['rl_variant']

    mpl = rl_variant['max_path_length']
    rl_variant['td3_trainer_kwargs']['discount'] = 1 - 1 / mpl
    rl_variant['sac_trainer_kwargs']['discount'] = 1 - 1 / mpl

    if args.debug:
        rl_variant['algo_kwargs']['num_epochs'] = 4
        rl_variant['algo_kwargs']['batch_size'] = 128
        rl_variant['contextual_replay_buffer_kwargs']['max_size'] = int(1e4)
        rl_variant['replay_buffer_kwargs']['max_size'] = int(1e4)
        rl_variant['algo_kwargs']['num_eval_steps_per_epoch'] = 200
        rl_variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 200
        rl_variant['algo_kwargs']['num_trains_per_train_loop'] = 200
        rl_variant['algo_kwargs']['min_num_steps_before_training'] = 200
        rl_variant['dump_video_kwargs']['columns'] = 2
        rl_variant['save_video_period'] = 2
        rl_variant['log_expl_video'] = False
        variant['imsize'] = 250
    rl_variant['renderer_kwargs']['img_width'] = variant['imsize']
    rl_variant['renderer_kwargs']['img_height'] = variant['imsize']

    if args.no_video:
        rl_variant['save_video'] = False

if __name__ == "__main__":
    args = parse_args()
    args.mem_per_exp = 7.0
    mount_blacklist = [
        'MountLocal@/home/soroush/research/bullet-manipulation',
        'MountLocal@/home/soroush/research/bullet-assets',
    ]
    preprocess_args(args)
    search_space = env_params[args.env]
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters(print_info=False)):
        process_variant(variant)
        run_experiment(
            exp_function=rl_experiment,
            variant=variant,
            args=args,
            exp_id=exp_id,
            mount_blacklist=mount_blacklist,
        )
