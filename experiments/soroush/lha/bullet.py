import rlkit.misc.hyperparameter as hyp
from exp_util import (
    run_experiment,
    parse_args,
    preprocess_args,
)
from rlkit.launchers.exp_launcher import rl_experiment
from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC

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
            eval_epoch_freq=25,
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
            # recompute_rewards=True,
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
        example_set_variant=dict(
            n=30,
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
                atomic=1.0,
                atomic_seq=0.0,
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
        'action_repeat': 10,
        'timestep': 1./120,
        'solver_iterations': 500,
        'max_force': 1000,

        'gui': False,
        'pos_init': [.75, -.3, 0],
        'pos_high': [.75, .4, .3],
        'pos_low': [.75, -.4, -.36],
        'reset_obj_in_hand_rate': 0.0,
        'bowl_bounds': [-0.40, 0.40],

        'use_rotated_gripper': True,
        'use_wide_gripper': True,
        'soft_clip': True,
        'obj_urdf': 'spam',
        'max_joint_velocity': None,

        'hand_reward': True,
        'gripper_reward': True,
        'bowl_reward': True,

        'goal_sampling_mode': 'ground',
        'random_init_bowl_pos': False,
        'bowl_type': 'fixed',
    },
    imsize=400,

    logger_config=dict(
        snapshot_gap=25,
        snapshot_mode='gap_and_last',
    ),
)

env_params = {
    'pb-4obj': {
        'env_kwargs.num_obj': [4],
        'env_kwargs.random_init_bowl_pos': [False],
        'env_kwargs.bowl_type': ['fixed'],

        'rl_variant.example_set_variant.subtask_codes': [
            [
                {2: 2, 3: 3},
                {4: 4, 5: 5},
                {6: 6, 7: 7},
                {8: 8, 9: 9},
            ],
        ],

        # 'rl_variant.mask_variant.mask_conditioned': [False],

        'rl_variant.mask_variant.mask_conditioned': [True],
        'rl_variant.mask_variant.param_variant.mask_format': ['cond_distribution'],
        'rl_variant.mask_variant.param_variant.infer_masks': [
            True,
            # False,
        ],
        'rl_variant.mask_variant.relabel_masks': [False],
        # 'rl_variant.mask_variant.relabel_goals': [False],

        'rl_variant.algo_kwargs.num_epochs': [4000],

        'rl_variant.mask_variant.eval_rollouts_to_log': [[]],

        # 'rl_variant.algo_kwargs.num_epochs': [2500],
        # 'rl_variant.algo_kwargs.eval_only': [True],
        # 'rl_variant.algo_kwargs.eval_epoch_freq': [100],
        # # 'rl_variant.algo_kwargs.num_eval_steps_per_epoch': [5000],
        # 'rl_variant.ckpt': [
        #     '/home/soroush/data/local/pb-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_46_02_id000--s13680',
        # ],
        # 'rl_variant.mask_variant.eval_mask_distr': [
        #     dict(
        #         atomic=0.0,
        #         atomic_seq=1.0,
        #         cumul_seq=0.0,
        #         full=0.0,
        #     ),
        # ]
    },
    'pb-4obj-rel': {
        'env_kwargs.num_obj': [4],
        'env_kwargs.random_init_bowl_pos': [True],
        'env_kwargs.bowl_type': ['heavy'],

        'rl_variant.expl_goal_sampling_mode': ['example_set'],
        'rl_variant.eval_goal_sampling_mode': ['obj_in_bowl'],

        'rl_variant.example_set_variant.subtask_codes': [
            [
                {2: -20, 3: 3},
            ],
        ],

        'rl_variant.mask_variant.mask_conditioned': [False],
        'rl_variant.contextual_mdp': [False],  # regular RL
        'env_kwargs.reward_type': ['bowl_cube0_dist'],

        # 'rl_variant.mask_variant.mask_conditioned': [True],
        # 'rl_variant.mask_variant.param_variant.mask_format': ['distribution'],
        # 'rl_variant.mask_variant.param_variant.infer_masks': [
        #     True,
        #     # False,
        # ],
        # 'rl_variant.contextual_replay_buffer_kwargs.fraction_future_context': [0.0], # no future relabeling

        'rl_variant.algo_kwargs.num_epochs': [5000],
        'rl_variant.mask_variant.eval_rollouts_to_log': [[]],
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

