import railrl.misc.hyperparameter as hyp
from railrl.misc.exp_util import (
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
        ),
        max_path_length=100,
        td3_trainer_kwargs=dict(
            use_policy_saturation_cost=True,
            policy_saturation_cost_threshold=5.0,
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
        algorithm="sac",
        context_based=True,
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
        save_video_period=50,
        renderer_kwargs=dict(),
        task_variant=dict(
            task_conditioned=False,
        ),
        mask_variant=dict(
            mask_conditioned=True,
            rollout_mask_order_for_expl='random',
            rollout_mask_order_for_eval='fixed',
            log_mask_diagnostics=True,
            mask_format='vector',
            infer_masks=False,
            mask_inference_variant=dict(
                n=1000,
                noise=0.01,
                normalize_sigma_inv=True,
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

            eval_rollouts_to_log=['atomic', 'atomic_seq', 'cumul_seq'],
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
        'goal_mode': 'obj_in_bowl',
        'num_obj': 0, #2

        'reward_type': 'obj_dist',

        # 'use_rotated_gripper': True,  # False
        # 'use_wide_gripper': False,  # False
        # 'soft_clip': True,
        # 'obj_urdf': 'spam_long',
        # 'max_joint_velocity': 1.0,

        'use_rotated_gripper': True,  # False
        'use_wide_gripper': True,  # False
        'soft_clip': True,
        'obj_urdf': 'spam',
        'max_joint_velocity': None,

        # 'use_rotated_gripper': False,  # False
        # 'use_wide_gripper': False,  # False
        # 'soft_clip': False,
        # 'obj_urdf': 'spam',
        # 'max_joint_velocity': None,
    },
    imsize=400,
)

env_params = {
    'pb-reach': {
        'env_kwargs.num_obj': [0],
        'rl_variant.algo_kwargs.num_epochs': [50],
        'rl_variant.save_video_period': [5],  # 25
    },
    'pb-1obj': {
        'env_kwargs.num_obj': [1],

        'env_kwargs.use_rotated_gripper': [
            # True,
            False,
        ],

        'rl_variant.algo_kwargs.num_epochs': [1000],
        'rl_variant.algo_kwargs.eval_epoch_freq': [10],
        'rl_variant.save_video_period': [100],
    },
    'pb-2obj': {
        'env_kwargs.num_obj': [2],
        # 'env_kwargs.reset_obj_in_hand_rate': [0.25],
        # 'env_kwargs.goal_mode': ['uniform_and_obj_in_bowl'],

        'env_kwargs.use_wide_gripper': [True],

        'rl_variant.mask_variant.idx_masks': [
            [
                {2: 2, 3: 3},
                {4: 4, 5: 5},
            ],
        ],

        'rl_variant.algo_kwargs.num_epochs': [3000],
        'rl_variant.algo_kwargs.eval_epoch_freq': [20],
        'rl_variant.save_video_period': [200],
    },
    'pb-4obj': {
        'env_kwargs.num_obj': [4],
        # 'rl_variant.max_path_length': [200],

        # 'rl_variant.mask_variant.idx_masks': [[
        #     {2: 2, 3: 3},
        #     {4: 4, 5: 5},
        #     {6: 6, 7: 7},
        #     {8: 8, 9: 9},
        # ]],

        'rl_variant.mask_variant.idx_masks': [[
            {0: -12, 1: -13},
            {2: 2, 3: 3},
            {0: -14, 1: -15},
            {4: 4, 5: 5},
            {0: -16, 1: -17},
            {6: 6, 7: 7},
            {0: -18, 1: -19},
            {8: 8, 9: 9},
        ]],
        'rl_variant.mask_variant.mask_groups': [[
            [0, 1], [2, 3], [4, 5], [6, 7],
        ]],

        'rl_variant.mask_variant.mask_format': ['distribution'],
        'rl_variant.mask_variant.infer_masks': [True],
        'rl_variant.mask_variant.mask_inference_variant.n': [
            50,
            # 1000,
        ],
        'rl_variant.mask_variant.eval_rollouts_to_log': [['atomic', 'atomic_seq']],

        # 'rl_variant.mask_variant.max_subtasks_per_rollout': [2],

        'rl_variant.algo_kwargs.num_epochs': [5000],
        'rl_variant.algo_kwargs.eval_epoch_freq': [20],
        'rl_variant.save_video_period': [200],
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
    rl_variant['renderer_kwargs']['img_width'] = variant['imsize']
    rl_variant['renderer_kwargs']['img_height'] = variant['imsize']
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

