import rlkit.misc.hyperparameter as hyp
from exp_util import (
    run_experiment,
    parse_args,
    preprocess_args,
)
from rlkit.launchers.exp_launcher import rl_experiment
from multiworld.envs.pygame.pick_and_place import PickAndPlaceEnv

from rlkit.envs.contextual.mask_conditioned import (
    default_masked_reward_fn,
    action_penalty_masked_reward_fn,
)

variant = dict(
    rl_variant=dict(
        do_state_exp=True,
        algo_kwargs=dict(
            num_epochs=1000,
            batch_size=2048,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000, #4000,
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
        expl_goal_sampling_mode='random',
        eval_goal_sampling_mode='random',
        algorithm="sac",
        context_based=True,
        save_env_in_snapshot=False,
        save_video=True,
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
            pad_color=0,
            pad_length=0,
            subpad_length=1,
        ),
        vis_kwargs=dict(
            vis_list=dict(),
        ),
        save_video_period=150,
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
        object_reward_only=False,

        init_position_strategy='random',
    ),
    imsize=256,

    logger_config=dict(
        snapshot_gap=50,
        snapshot_mode='gap_and_last',
    ),
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
        'rl_variant.mask_variant.idx_masks': [
            [
                {2: 2, 3: 3},
                {4: 4, 5: 5},
                {6: 6, 7: 7},
                {8: 8, 9: 9},
            ],
        ],

        'rl_variant.algo_kwargs.num_epochs': [4000],
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
    args.mem_per_exp = 3.5
    mount_blacklist = [
        'MountLocal@/home/soroush/research/furniture',
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

