import railrl.misc.hyperparameter as hyp
from railrl.misc.exp_util import (
    run_experiment,
    parse_args,
    preprocess_args,
)
from railrl.launchers.exp_launcher import rl_experiment
from roboverse.envs.goal_conditioned.sawyer_lift import SawyerLiftEnvGC

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

        'use_wide_gripper': True,
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
        'env_kwargs.reward_type': [
            'obj_dist',
        ],

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
        'env_kwargs.reward_type': [
            'obj_dist',
        ],

        'env_kwargs.use_rotated_gripper': [
            True,
            False,
        ],

        'rl_variant.algo_kwargs.num_epochs': [3000],
        'rl_variant.algo_kwargs.eval_epoch_freq': [10],
        'rl_variant.save_video_period': [150],
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

