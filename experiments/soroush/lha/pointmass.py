import railrl.misc.hyperparameter as hyp
from railrl.misc.exp_util import (
    run_experiment,
    parse_args,
    preprocess_args,
)
from railrl.launchers.exp_launcher import rl_experiment
from multiworld.envs.pygame.pick_and_place import PickAndPlaceEnv

from railrl.launchers.contextual.state_based import (
    default_masked_reward_fn,
    action_penalty_masked_reward_fn,
)

variant = dict(
    rl_variant=dict(
        do_state_exp=True,
        algo_kwargs=dict(
            num_epochs=1000,
            batch_size=1024,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000, #4000,
            min_num_steps_before_training=1000,
        ),
        max_path_length=100,
        td3_trainer_kwargs=dict(
            use_policy_saturation_cost=True,
            policy_saturation_cost_threshold=5.0,
            reward_scale=10,
        ),
        sac_trainer_kwargs=dict(
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale=100,
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
        algorithm="sac",
        context_based=True,
        save_video=True,
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
            pad_color=0,
            pad_length=0,
            subpad_length=1,
        ),
        save_video_period=150,
        renderer_kwargs=dict(),
        goal_sampling_mode='random',
        task_variant=dict(
            task_conditioned=False,
        ),
        mask_variant=dict(
            mask_conditioned=True,
            rollout_mask_order_for_expl='fixed',
            rollout_mask_order_for_eval='fixed',
            log_mask_diagnostics=True,
            mask_format='vector',
            infer_masks=False,
            mask_inference_variant=dict(
                n=1000,
                noise=0.1,
                normalize_sigma_inv=True,
            ),
            relabel_goals=True,
            relabel_masks=True,
            sample_masks_for_relabeling=True,

            context_post_process_mode=None,
            context_post_process_frac=0.5,

            max_subtasks_to_focus_on=None,
            prev_subtask_weight=1.0,
            reward_fn=default_masked_reward_fn,

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
        ),
    ),
    # env_id='FourObject-PickAndPlace-RandomInit-2D-v1',
    env_class=PickAndPlaceEnv,
    env_kwargs=dict(
        # Environment dynamics
        action_scale=1.0,
        ball_radius=0.75, #1.
        boundary_dist=4,
        object_radius=0.50,
        min_grab_distance=0.5,
        walls=None,
        # Rewards
        action_l2norm_penalty=0,
        reward_type="dense", #dense_l1
        success_threshold=0.60,
        # Reset settings
        fixed_goal=None,
        # Visualization settings
        images_are_rgb=True,
        render_dt_msec=0,
        render_onscreen=False,
        render_size=84,
        show_goal=True,
        # get_image_base_render_size=(48, 48),
        # Goal sampling
        goal_samplers=None,
        goal_sampling_mode='random',
        num_presampled_goals=10000,
        object_reward_only=True,

        init_position_strategy='random',
    ),
    imsize=256,
)

env_params = {
    'pg-1obj': {
        'env_kwargs.num_objects': [1],
        'env_kwargs.object_reward_only': [
            True,
            False,
        ],
        'rl_variant.algo_kwargs.num_epochs': [500],
        'rl_variant.save_video_period': [50],
    },
    'pg-2obj': {
        'env_kwargs.num_objects': [2],
        'rl_variant.algo_kwargs.num_epochs': [500],

        'rl_variant.mask_variant.mask_conditioned': [True],
        'rl_variant.mask_variant.mask_idxs': [
            [[0, 1], [2, 3], [4, 5]],
            [[2, 3, 4, 5]],
        ],
    },
    'pg-4obj': {
        'env_kwargs.num_objects': [4],
        'rl_variant.algo_kwargs.num_epochs': [6000],

        # 'rl_variant.mask_variant.max_subtasks_to_focus_on': [2],
        # 'rl_variant.mask_variant.reward_fn': [action_penalty_masked_reward_fn],

        'rl_variant.mask_variant.prev_subtask_weight': [0.15],

        'rl_variant.mask_variant.idx_masks': [
            [
                {2: 2, 3: 3},
                {4: 4, 5: 5},
                {6: 6, 7: 7},
                {8: 8, 9: 9},
            ],
        ],

        'rl_variant.mask_variant.train_mask_distr': [
            dict(
                atomic=0.5,
                cumul=0.5,
                subset=0.0,
                full=0.0,
            ),
        ],

        'rl_variant.mask_variant.expl_mask_distr': [
            dict(
                atomic=0.5,
                atomic_seq=0.5,
                cumul_seq=0.0,
                full=0.0,
            ),
            # dict(
            #     atomic=0.3,
            #     atomic_seq=0.3,
            #     cumul_seq=0.4,
            #     full=0.0,
            # ),
        ],

        # 'rl_variant.save_video_period': [4],


        # 'rl_variant.ckpt': [
        #     'pg-4obj/06-06-larger-nupo/06-06-larger-nupo_2020_06_07_03_24_58_id000--s75821',
        #     'pg-4obj/06-06-larger-nupo/06-06-larger-nupo_2020_06_07_03_24_59_id000--s96595',
        #     'pg-4obj/06-06-larger-nupo/06-06-larger-nupo_2020_06_07_03_24_58_id000--s16595',
        # ],
        # 'rl_variant.algo_kwargs.do_training': [False],
        # 'rl_variant.use_sampling_policy': [
        #     False,
        #     # True,
        # ],
        # 'rl_variant.mask_variant.prev_subtasks_solved': [
        #     # True,
        #     False,
        # ],
        # 'rl_variant.dump_video_kwargs.keys_to_show': [[
        #     # 'image_v',
        #     # 'image_v_1',
        #     # 'image_v_2',
        #     # 'image_v_3',
        #     # 'image_v_4',
        #
        #     'image_pi',
        #     'image_pi_0',
        #     # 'image_pi_1',
        #     # 'image_pi_2',
        #     # 'image_pi_3',
        #     # 'image_pi_4',
        # ]],
        # 'rl_variant.log_expl_video': [False],
        # 'rl_variant.algo_kwargs.num_epochs': [3],
        # 'rl_variant.save_video_period': [1],
        # 'rl_variant.dump_video_kwargs.columns': [4],
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
        rl_variant['algo_kwargs']['num_eval_steps_per_epoch'] = 200
        rl_variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 200
        rl_variant['algo_kwargs']['num_trains_per_train_loop'] = 200
        rl_variant['algo_kwargs']['min_num_steps_before_training'] = 200
        rl_variant['dump_video_kwargs']['columns'] = 2
        rl_variant['save_video_period'] = 2
        rl_variant['log_expl_video'] = False
        variant['imsize'] = 256
    rl_variant['renderer_kwargs']['img_width'] = variant['imsize']
    rl_variant['renderer_kwargs']['img_height'] = variant['imsize']

    if args.no_video:
        rl_variant['save_video'] = False

if __name__ == "__main__":
    args = parse_args()
    args.mem_per_exp = 3.5
    mount_blacklist = [
        'MountLocal@/home/soroush/research/furniture',
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
            snapshot_mode='gap_and_last',
        )

