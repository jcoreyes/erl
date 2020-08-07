import argparse
import math
import numpy as np

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.contextual.state_based_steven_masking import (
    masking_sac_experiment_with_variant
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.contextual.state_based_steven_masking import (
     PickAndPlace1DEnvObjectOnlyRewardFn
)

from multiworld.envs.pygame.pick_and_place import (
    PickAndPlaceEnv,
    PickAndPlace1DEnv,
)

if __name__ == "__main__":
    variant = dict(
        # train_env_id='TwoObjectPickAndPlaceRandomInit1DEnv-v0',
        # eval_env_id='TwoObjectPickAndPlaceRandomInit1DEnv-v0',
        # train_env_id='FourObjectPickAndPlaceRandomInit2DEnv-v0',
        # eval_env_id='FourObjectPickAndPlaceRandomInit2DEnv-v0',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        sac_trainer_kwargs=dict(
            reward_scale=100,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=300,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='ou',
            exploration_noise=0.0, #0.3
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.4,
            fraction_distribution_context=0.4,
            max_size=int(1e6),
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=50, #10
            pad_color=0,
            rows=1,
            columns=8,
        ),
        renderer_kwargs=dict(
            img_width=256,
            img_height=256,
        ),
        exploration_goal_sampling_mode='random',
        evaluation_goal_sampling_mode='random',
        do_masking=True,
        masking_reward_fn=PickAndPlace1DEnvObjectOnlyRewardFn,
        mask_dim=2,
        log_mask_diagnostics=True, #False
        masking_eval_steps=300,
        rotate_masks_for_eval=True,
        rotate_masks_for_expl=True,
        masking_for_exploration=True,
        mask_distribution='one_hot_masks',
        # exploration_policy_path='/home/steven/logs/20-04-27-train-expert-policy-gpu/20-04-27-train-expert-policy-gpu_2020_04_27_00_53_47_id205903--s471039/params.pkl',
        env_class=PickAndPlaceEnv,
        env_kwargs=dict(
            # Environment dynamics
            action_scale=1.0,
            ball_radius=0.75,  # 1.
            boundary_dist=4,
            object_radius=0.50,
            min_grab_distance=0.5,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense",  # dense_l1
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
    )

    env_params = {
        'pg-4obj': {
            'env_kwargs.num_objects': [4],
            'algo_kwargs.num_epochs': [1500],
            'mask_dim': [4],
        },
    }


    def process_variant(variant):
        mpl = variant['max_path_length']
        variant['sac_trainer_kwargs']['discount'] = 1 - 1 / mpl

        if args.debug:
            variant['algo_kwargs']['num_epochs'] = 4
            variant['algo_kwargs']['batch_size'] = 128
            variant['replay_buffer_kwargs']['max_size'] = int(1e4)
            variant['algo_kwargs']['num_eval_steps_per_epoch'] = 200
            variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 200
            variant['algo_kwargs']['num_trains_per_train_loop'] = 200
            variant['algo_kwargs']['min_num_steps_before_training'] = 200
            variant['save_video_kwargs']['save_video_period'] = 2

        if args.no_video:
            variant['save_video'] = False


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='block-2obj'),
        parser.add_argument('--mode', type=str, default='local')
        parser.add_argument('--label', type=str, default='')
        parser.add_argument('--num_seeds', type=int, default=1)
        parser.add_argument('--max_exps_per_instance', type=int, default=2)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--first_variant_only', action='store_true')
        parser.add_argument('--no_video', action='store_true')
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
                    masking_sac_experiment_with_variant,
                    exp_folder=args.env,
                    exp_prefix=exp_prefix,
                    exp_id=exp_id,
                    mode=args.mode,
                    variant=variant,
                    use_gpu=(not args.no_gpu),
                    gpu_id=args.gpu_id,

                    num_exps_per_instance=int(num_exps),

                    snapshot_gap=50,
                    snapshot_mode="none",  # 'gap_and_last',
                )

                if args.first_variant_only:
                    exit()