import railrl.misc.hyperparameter as hyp
from exp_util import (
    run_experiment,
    parse_args,
    preprocess_args,
)
from railrl.launchers.exp_launcher import rl_experiment
from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC

from railrl.envs.contextual.mask_conditioned import (
    default_masked_reward_fn,
    action_penalty_masked_reward_fn,
)

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
            eval_epoch_freq=20,
        ),
        max_path_length=100,
        td3_trainer_kwargs=dict(
            discount=0.99,
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
            recompute_rewards=True,
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
        expl_goal_sampling_mode='50p_ground__50p_obj_in_bowl',
        eval_goal_sampling_mode='obj_in_bowl',
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
        save_video_period=200,
        renderer_kwargs=dict(),
        task_variant=dict(
            task_conditioned=False,
        ),
        mask_variant=dict(
            mask_conditioned=True,
            rollout_mask_order_for_expl='random',
            rollout_mask_order_for_eval='fixed',
            log_mask_diagnostics=True,
            mask_format='matrix',
            infer_masks=False,
            mask_inference_variant=dict(
                n=100,
                noise=0.01,
                max_cond_num=1e2,
                normalize_sigma_inv=True,
                sigma_inv_entry_threshold=0.10,
            ),
            relabel_goals=True,
            relabel_masks=True,
            sample_masks_for_relabeling=True,

            context_post_process_mode=None,
            context_post_process_frac=0.5,

            max_subtasks_to_focus_on=None,
            max_subtasks_per_rollout=None,
            prev_subtask_weight=0.25,
            reward_fn=default_masked_reward_fn,
            use_g_for_mean=True,

            train_mask_distr=dict(
                atomic=1.0,
                subset=0.0,
                cumul=0.0,
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
    ),
    env_class=SawyerLiftEnvGC,
    env_kwargs={
        'action_scale': .06,
        'action_repeat': 10, #5
        'timestep': 1./120, #1./240
        'solver_iterations': 500, #150
        'max_force': 1000,

        'gui': False,
        'pos_init': [.75, -.3, 0],
        'pos_high': [.75, .4, .3],
        'pos_low': [.75, -.4, -.36],
        'reset_obj_in_hand_rate': 0.0,
        'goal_sampling_mode': 'ground',
        'random_init_bowl_pos': False,
        'sliding_bowl': False,
        'heavy_bowl': False,
        'bowl_bounds': [-0.40, 0.40],

        'reward_type': 'obj_dist',

        'use_rotated_gripper': True,  # False
        'use_wide_gripper': True,  # False
        'soft_clip': True,
        'obj_urdf': 'spam',
        'max_joint_velocity': None,
    },
    imsize=400,

    logger_config=dict(
        snapshot_gap=50,
        snapshot_mode='gap_and_last',
    ),
)

env_params = {
    'pb-reach': {
        'env_kwargs.num_obj': [0],
        'env_kwargs.sliding_bowl': [True],
        'env_kwargs.heavy_bowl': [False],
        'env_kwargs.random_init_bowl_pos': [True],
        'rl_variant.mask_variant.mask_conditioned': [False],
        'rl_variant.algo_kwargs.num_epochs': [50],

        'env_kwargs.reward_type': ['hand_dist+obj_dist'],

        'rl_variant.save_video_period': [5],
        'rl_variant.dump_video_kwargs.columns': [3],
        'rl_variant.algo_kwargs.eval_epoch_freq': [1],

    },
    'pb-1obj': {
        'env_kwargs.num_obj': [1],
        'env_kwargs.sliding_bowl': [True],
        'env_kwargs.heavy_bowl': [True],
        'env_kwargs.random_init_bowl_pos': [True],

        # 'rl_variant.mask_variant.idx_masks': [
        #     [
        #         {2: -14, 3: -13},
        #     ],
        # ],

        'rl_variant.mask_variant.matrix_masks': [
            [[
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, -1],
                [0, 0, 0, 1, 0],
                [0, 0, -1, 0, 1],
            ]],
            # [[
            #     [0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0],
            #     [0, 0, 1, 0, 0],
            #     [0, 0, 0, 1, 0],
            #     [0, 0, 0, 0, 1],
            # ]],
        ],

        'rl_variant.algo_kwargs.num_epochs': [2500],
    },
    'pb-2obj': {
        'env_kwargs.num_obj': [2],
        
        'rl_variant.mask_variant.idx_masks': [
            [
                {2: 2, 3: 3},
                {4: 4, 5: 5},
            ],
        ],
        # 'rl_variant.mask_variant.matrix_masks': [
        #     [[
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 1, 0, 0, 0],
        #         [0, 0, 0, 1, 0, 0],
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0],
        #     ],
        #     [
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 1, 0],
        #         [0, 0, 0, 0, 0, 1],
        #     ]],
        #
        #     # [[
        #     #     [0.01, 0, 0, 0, 0, 0],
        #     #     [0, 0.01, 0, 0, 0, 0],
        #     #     [0, 0, 0.99, 0, 0, 0],
        #     #     [0, 0, 0, 1.00, 0, 0],
        #     #     [0, 0, 0, 0, 0.01, 0],
        #     #     [0, 0, 0, 0, 0, 0.01],
        #     # ],
        #     # [
        #     #     [0.01, 0, 0, 0, 0, 0],
        #     #     [0, 0.01, 0, 0, 0, 0],
        #     #     [0, 0, 0.01, 0, 0, 0],
        #     #     [0, 0, 0, 0.01, 0, 0],
        #     #     [0, 0, 0, 0, 0.98, 0],
        #     #     [0, 0, 0, 0, 0, 1.00],
        #     # ]],
        # ],

        'rl_variant.expl_goal_sampling_mode': ['ground'],

        'rl_variant.mask_variant.mask_format': [
            'distribution',
            # 'matrix',
            # 'vector',
        ],
        'rl_variant.mask_variant.infer_masks': [True],
        'rl_variant.mask_variant.mask_inference_variant.sigma_inv_entry_threshold': [
            # 0.10,
            None,
        ],

        'rl_variant.algo_kwargs.num_epochs': [2500],
    },
    'pb-3obj': {
        'env_kwargs.num_obj': [3],
        'env_kwargs.sliding_bowl': [True],
        'env_kwargs.random_init_bowl_pos': [True],

        # 'rl_variant.mask_variant.idx_masks': [
        #     [
        #         {2: -14, 3: -13},
        #     ],
        # ],

        'rl_variant.mask_variant.matrix_masks': [
            # [[
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            #     [0, 0, 1, 0, 0, 0, 0, 0, -1],
            #     [0, 0, 0, 1, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            #     [0, 0, -1, 0, 0, 0, 0, 0, 1],
            # ]],
            [[
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]],
        ],

        'rl_variant.mask_variant.mask_format': [
            # 'distribution',
            'matrix',
        ],

        # 'rl_variant.mask_variant.mask_conditioned': [
        #     # True,
        #     False,
        # ],
        # 'rl_variant.mask_variant.mask_format': ['distribution'],
        # 'rl_variant.mask_variant.infer_masks': [True],
        # 'rl_variant.mask_variant.mask_inference_variant.n': [
        #     # 50,
        #     1000,
        # ],

        'rl_variant.algo_kwargs.num_epochs': [6000],
    },
    'pb-4obj': {
        'env_kwargs.num_obj': [4],
        # 'rl_variant.max_path_length': [200],

        'rl_variant.mask_variant.idx_masks': [[
            {2: 2, 3: 3},
            {4: 4, 5: 5},
            {6: 6, 7: 7},
            {8: 8, 9: 9},
        ]],

        # 'rl_variant.mask_variant.eval_rollouts_to_log': [['atomic_seq']],

        'env_kwargs.sliding_bowl': [True],
        'env_kwargs.heavy_bowl': [True],
        'env_kwargs.random_init_bowl_pos': [True],

        # 'rl_variant.mask_variant.idx_masks': [[
        #     {0: -12, 1: -13},
        #     {2: 2, 3: 3},
        #     {0: -14, 1: -15},
        #     {4: 4, 5: 5},
        #     {0: -16, 1: -17},
        #     {6: 6, 7: 7},
        #     {0: -18, 1: -19},
        #     {8: 8, 9: 9},
        # ]],
        # 'rl_variant.mask_variant.mask_groups': [[
        #     [0, 1], [2, 3], [4, 5], [6, 7],
        # ]],

        'rl_variant.mask_variant.mask_conditioned': [
            True,
            # False,
        ],

        # 'rl_variant.mask_variant.mask_format': ['distribution'],
        # 'rl_variant.mask_variant.infer_masks': [True],
        # 'rl_variant.mask_variant.mask_inference_variant.n': [
        #     # 50,
        #     1000,
        # ],

        'rl_variant.algo_kwargs.num_epochs': [6000],
    },
    'pb-5obj': {
        'env_kwargs.num_obj': [5],

        'rl_variant.max_path_length': [150],
        'rl_variant.algo_kwargs.num_eval_steps_per_epoch': [1500],
        'rl_variant.algo_kwargs.num_expl_steps_per_train_loop': [1500],
        'rl_variant.algo_kwargs.num_trains_per_train_loop': [1500],
        'rl_variant.algo_kwargs.min_num_steps_before_training': [1500],

        'rl_variant.mask_variant.idx_masks': [
            # [
            #     {2: 2, 3: 3},
            #     {4: 4, 5: 5},
            #     {6: 6, 7: 7},
            #     {8: 8, 9: 9},
            #     {10: 10, 11: 11},
            # ],

            [
                {i:i for i in range(12)}
            ],
        ],

        # 'rl_variant.mask_variant.infer_masks': [True],
        # 'rl_variant.mask_variant.mask_inference_variant.n': [
        #     # 50,
        #     1000,
        # ],

        'rl_variant.algo_kwargs.num_epochs': [8000],
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
        # rl_variant['log_expl_video'] = False
        variant['imsize'] = 256
    rl_variant['renderer_kwargs']['width'] = variant['imsize']
    rl_variant['renderer_kwargs']['height'] = variant['imsize']
    variant['env_kwargs']['img_dim'] = variant['imsize']

    if args.no_video:
        rl_variant['save_video'] = False

if __name__ == "__main__":
    args = parse_args()
    args.mem_per_exp = 5.0
    mount_blacklist = [
        'MountLocal@/home/soroush/research/furniture',
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

