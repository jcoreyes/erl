from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp
import argparse
import math
import numpy as np

from railrl.launchers.exp_launcher import rl_experiment

from multiworld.envs.pygame.pick_and_place import (
    PickAndPlaceEnv,
    PickAndPlace1DEnv,
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
        use_subgoal_policy=False,
        subgoal_policy_kwargs=dict(
            num_subgoals_per_episode=2,
        ),
        use_masks=False,
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
        vis_kwargs=dict(
            vis_list=dict(),
        ),
        save_video_period=50,
        renderer_kwargs=dict(),
        goal_sampling_mode='random',
        task_variant=dict(
            task_conditioned=False,
        ),
        mask_variant=dict(
            mask_conditioned=False,
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

            train_mask_distr=dict(
                atomic=0.6,
                cumul=0.3,
                full=0.1,
            ),
            expl_mask_distr=dict(
                atomic=0.6,
                cumul_seq=0.3,
                full=0.1,
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
        'rl_variant.algo_kwargs.num_epochs': [1500],

        'rl_variant.mask_variant.mask_conditioned': [True],
        'rl_variant.mask_variant.idx_masks': [
            [
                {2: 2, 3: 3},
                {4: 4, 5: 5},
                {6: 6, 7: 7},
                {8: 8, 9: 9},
            ],
            # [
            #     {0: 0, 1: 1},
            #     {2: 2, 3: 3},
            #     {4: 4, 5: 5},
            #     {6: 6, 7: 7},
            #     {8: 8, 9: 9},
            # ],
            # [
            #     {2: 2, 3: 3},
            #     {4: 4, 5: 5},
            #     {6: 6, 7: 7},
            #     {8: 8, 9: 9},
            #     {0: 0, 1: 1},
            # ],
            # [
            #     {0: 0, 1: 1},
            #     {2: 2, 3: 3},
            #     {4: 4, 5: 5},
            #     {6: 6, 7: 7},
            #     {8: 8, 9: 9},
            #     {0: -12, 1: -13},
            #     {0: -14, 1: -15},
            #     {0: -16, 1: -17},
            #     {0: -18, 1: -19},
            # ],
            # [
            #     {0: 0, 1: 1},
            #     {0: -12, 1: -13},
            #     {0: -14, 1: -15},
            #     {0: -16, 1: -17},
            #     {0: -18, 1: -19},
            # ],
        ],

        'rl_variant.mask_variant.train_mask_distr': [
            dict(
                atomic=1.0,
                cumul=0.0,
                full=0.0,
            ),
            dict(
                atomic=0.6,
                cumul=0.3,
                full=0.1,
            ),
        ],

        'rl_variant.mask_variant.expl_mask_distr': [
            dict(
                atomic=1.0,
                cumul_seq=0.0,
                full=0.0,
            ),
            dict(
                atomic=0.6,
                cumul_seq=0.3,
                full=0.1,
            ),
        ],
    },
    # 'pg-4obj': {
    #     'env_kwargs.num_objects': [4],
    #     'rl_variant.algo_kwargs.num_epochs': [2500],
    #     'rl_variant.max_path_length': [200],
    #
    #     'rl_variant.mask_variant.mask_conditioned': [True],
    #     'rl_variant.mask_variant.idx_masks': [
    #         # [
    #         #     {0: 0, 1: 1},
    #         #     {2: 2, 3: 3},
    #         #     {4: 4, 5: 5},
    #         #     {6: 6, 7: 7},
    #         #     {8: 8, 9: 9},
    #         # ],
    #         [
    #             {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
    #         ],
    #     ],
    # },
    # 'pg-6obj': {
    #     'env_kwargs.num_objects': [6],
    #     'rl_variant.algo_kwargs.num_epochs': [2500],
    #     'rl_variant.max_path_length': [200],
    #
    #     'rl_variant.mask_variant.mask_conditioned': [True],
    #     'rl_variant.mask_variant.idx_masks': [
    #         [
    #             {0: 0, 1: 1},
    #             {2: 2, 3: 3},
    #             {4: 4, 5: 5},
    #             {6: 6, 7: 7},
    #             {8: 8, 9: 9},
    #             # {10: 10, 11: 11},
    #             # {12: 12, 13: 13},
    #         ],
    #     ],
    # },
    # 'pg-8obj': {
    #     'env_kwargs.num_objects': [8],
    #     'rl_variant.algo_kwargs.num_epochs': [2500],
    #     'rl_variant.max_path_length': [200],
    #
    #     'rl_variant.mask_variant.mask_conditioned': [True],
    #     'rl_variant.mask_variant.idx_masks': [
    #         [
    #             {0: 0, 1: 1},
    #             {2: 2, 3: 3},
    #             {4: 4, 5: 5},
    #             {6: 6, 7: 7},
    #             {8: 8, 9: 9},
    #             # {10: 10, 11: 11},
    #             # {12: 12, 13: 13},
    #             # {14: 14, 15: 15},
    #             # {16: 16, 17: 17},
    #         ],
    #     ],
    # },
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
                exp_id=exp_id,
                mode=args.mode,
                variant=variant,
                use_gpu=(not args.no_gpu),
                gpu_id=args.gpu_id,

                num_exps_per_instance=int(num_exps),

                snapshot_gap=50,
                snapshot_mode="none", #'gap_and_last',
            )

            if args.first_variant_only:
                exit()

