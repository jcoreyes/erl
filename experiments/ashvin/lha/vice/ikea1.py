from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.experiments.goal_distribution.irl_launcher import \
    representation_learning_with_goal_distribution_launcher

from rlkit.launchers.launcher_util import run_experiment
# from rlkit.torch.sets.launcher import test_offline_set_vae
# from rlkit.launchers.masking_launcher import default_masked_reward_fn
from rlkit.envs.contextual.mask_conditioned import default_masked_reward_fn
from rlkit.launchers.exp_launcher import rl_context_experiment

from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC
from furniture.env.furniture_multiworld import FurnitureMultiworld

variant = dict(
    rl_variant=dict(
        do_state_exp=True,
        algo_kwargs=dict(
            num_epochs=1000,
            batch_size=2048,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,  # 4000,
            min_num_steps_before_training=1000,
            eval_epoch_freq=1,
        ),
        max_path_length=100,
        td3_trainer_kwargs=dict(
            discount=0.99,
            reward_scale=10,
        ),
        sac_trainer_kwargs=dict(
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale=100,
            discount=0.99,
        ),
        contextual_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_future_context=0.4,
            fraction_distribution_context=0.4,
            fraction_replay_buffer_context=0.0,
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
        exploration_type='ou',
        exploration_noise=0.3,
        expl_goal_sampling_mode='assembled',
        eval_goal_sampling_mode='assembled',
        algorithm="sac",
        context_based=True,
        save_env_in_snapshot=False,
        save_video=True,
        dump_video_kwargs=dict(
            rows=1,
            columns=6,
            pad_color=0,
            pad_length=0,
            subpad_length=1,
        ),
        vis_kwargs=dict(
            vis_list=dict(),
        ),
        save_video_period=150,
        renderer_kwargs=dict(),
        example_set_variant=dict(
            n=50,
            subtask_codes=None,
            other_dims_random=True,
            use_cache=False,
            cache_path=None,
        ),
        mask_variant=dict(
            mask_conditioned=True,
            rollout_mask_order_for_expl='random',
            rollout_mask_order_for_eval='fixed',
            log_mask_diagnostics=True,
            param_variant=dict(
                mask_format='matrix',
                infer_masks=False,
                noise=0.01,
                max_cond_num=1e2,
                normalize_mask=True,
                mask_threshold=0.25,
            ),
            relabel_goals=True,
            relabel_masks=True,
            sample_masks_for_relabeling=True,

            context_post_process_mode=None,
            context_post_process_frac=0.5,

            max_subtasks_to_focus_on=None,
            max_subtasks_per_rollout=None,
            prev_subtask_weight=0.25,
            use_g_for_mean=False,

            train_mask_distr=dict(
                atomic=1.0,
                subset=0.0,
                full=0.0,
            ),
            expl_mask_distr=dict(
                atomic=0.5,
                atomic_seq=0.5,
                cumul_seq=0.0,
                full=0.0,
            ),
            eval_mask_distr=dict(
                atomic=0.0,
                atomic_seq=1.0,
                cumul_seq=0.0,
                full=0.0,
            ),

            eval_rollouts_to_log=['atomic', 'atomic_seq'],
            eval_rollouts_for_videos=[],
        ),
        keep_env_infos=True,
    ),
    env_class=FurnitureMultiworld,
    env_kwargs=dict(
        name="FurnitureCursorRLEnv",
        unity=False,
        tight_action_space=True,
        preempt_collisions=True,
        boundary=[0.5, 0.5, 0.95],
        pos_dist=0.2,
        num_connect_steps=0,
        num_connected_ob=False,
        num_connected_reward_scale=5.0,
        goal_type='zeros', #reset
        reset_type='var_2dpos+no_rot', #'var_2dpos+var_1drot', 'var_2dpos+objs_near',

        task_type='connect',
        control_degrees='3dpos+select+connect',
        obj_joint_type='slide',
        connector_ob_type=None, #'dist',

        clip_action_on_collision=True,

        light_logging=True,
    ),
    imsize=256,

    logger_config=dict(
        snapshot_gap=25,
        snapshot_mode='gap_and_last',
    ),
)

env_params = {
    'block-2obj': {
        'env_kwargs.furniture_name': ['block'],
        'env_kwargs.reward_type': [
            'object_distance',
        ],

        'rl_variant.algo_kwargs.num_epochs': [2000],
        'rl_variant.save_video_period': [50],  # 50

        'rl_variant.mask_variant.mask_conditioned': [False],

        'env_kwargs.task_type': [
            # 'connect',
            # 'select+connect',
            # 'reach+select+connect',
            'reach2+select+connect',
        ],
    },
    'shelf-4obj': {
        'env_kwargs.furniture_name': ['shelf_ivar_0678_4obj_bb'],
        'env_kwargs.anchor_objects': [['1_column']],
        'env_kwargs.reward_type': [
            'state_distance',
        ],
        'env_kwargs.move_speed': [0.05],
        'env_kwargs.goal_sampling_mode': ['uniform'],

        # 'env_kwargs.task_type': [
        #     # 'move2',
        #     'select2+move2',
        #     # 'reach2+select2+move2',
        # ],
        # 'rl_variant.expl_goal_sampling_mode' : ['assembled'],
        # 'rl_variant.eval_goal_sampling_mode' : ['assembled'],
        # 'rl_variant.example_set_variant.subtask_codes': [
        #     [
        #         {17: 17, 18: 18, 19: 19},
        #         {14: 14, 15: 15, 16: 16},
        #         {11: 11, 12: 12, 13: 13},
        #     ],
        # ],
        # # 'rl_variant.example_set_variant.subtask_codes': [
        # #     [
        # #         {17: 17, 18: 18, 19: 19},
        # #     ],
        # # ],

        'env_kwargs.task_type': ['select2'],
        'rl_variant.expl_goal_sampling_mode': ['assembled_random'],
        'rl_variant.eval_goal_sampling_mode': ['assembled_random'],
        'rl_variant.example_set_variant.subtask_codes': [
            [
                {8: 8, 9: 9, 10: 10, 17: 17, 18: 18, 19: 19},
                {8: 8, 9: 9, 10: 10, 14: 14, 15: 15, 16: 16},
                {8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13},
            ],
        ],
        # 'rl_variant.example_set_variant.subtask_codes': [
        #     [
        #         {8: 8, 9: 9, 10: 10, 17: 17, 18: 18, 19: 19},
        #     ],
        # ],

        # 'rl_variant.mask_variant.mask_conditioned': [False],

        'rl_variant.mask_variant.mask_conditioned': [True],
        'rl_variant.mask_variant.param_variant.mask_format': ['cond_distribution'],
        'rl_variant.mask_variant.param_variant.infer_masks': [
            True,
            # False,
        ],

        'rl_variant.algo_kwargs.eval_epoch_freq': [25],
        'rl_variant.algo_kwargs.num_epochs': [5000], #500
        'rl_variant.save_video_period': [100],  # 25
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

    if args.debug:
        rl_variant['algo_kwargs']['num_epochs'] = 4
        rl_variant['algo_kwargs']['batch_size'] = 128
        rl_variant['contextual_replay_buffer_kwargs']['max_size'] = int(1e4)
        rl_variant['algo_kwargs']['num_eval_steps_per_epoch'] = 200
        rl_variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 200
        rl_variant['algo_kwargs']['num_trains_per_train_loop'] = 200
        rl_variant['algo_kwargs']['min_num_steps_before_training'] = 200
        rl_variant['dump_video_kwargs']['columns'] = 2
        rl_variant['save_video_period'] = 2
        rl_variant['log_expl_video'] = False
        variant['imsize'] = 256
    rl_variant['renderer_kwargs']['width'] = variant['imsize']
    rl_variant['renderer_kwargs']['height'] = variant['imsize']

    if args.no_video:
        rl_variant['save_video'] = False

if __name__ == "__main__":
    args = parse_args()
    args.mem_per_exp = 7.0
    mount_blacklist = [
        'MountLocal@/home/soroush/research/bullet-manipulation',
    ]
    preprocess_args(args)
    search_space = env_params[args.env]
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters(verbose=False)):
        process_variant(variant)
        variant['exp_id'] = exp_id
        run_experiment(
            exp_function=rl_experiment,
            variant=variant,
            args=args,
            exp_id=exp_id,
            mount_blacklist=mount_blacklist,
        )
