import rlkit.misc.hyperparameter as hyp
from exp_util import (
    run_experiment,
    parse_args,
    preprocess_args,
)
from rlkit.launchers.exp_launcher import rl_experiment

from multiworld.envs.pygame.pick_and_place import PickAndPlaceEnv
from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC
from furniture.env.furniture_multiworld import FurnitureMultiworld

variant = dict(
    rl_variant=dict(
        do_state_exp=True,
        num_rollouts_per_epoch=15,
        algo_kwargs=dict(
            num_epochs=1000,
            batch_size=2048,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000, #4000,
            min_num_steps_before_training=1000,

            eval_only=True,
            eval_epoch_freq=50,
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
            max_size=int(1E4),
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
        expl_goal_sampling_mode=None,
        eval_goal_sampling_mode=None,
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
        save_video_period=400,
        renderer_kwargs=dict(),
        example_set_variant=dict(
            # n=30,
            # subtask_codes=None,
            # other_dims_random=True,
            # use_cache=False,
            # cache_path=None,
        ),
        mask_variant=dict(
            mask_conditioned=True,
            rollout_mask_order_for_expl='random',
            rollout_mask_order_for_eval='fixed',
            log_mask_diagnostics=False,
            param_variant=dict(
                mask_format='cond_distribution',
                # infer_masks=False,
                # noise=0.10,
                # max_cond_num=1e2,
                # normalize_mask=True,
                # mask_threshold=0.25,
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

            ### change one of these values later! ###
            eval_mask_distr=dict(
                atomic=0.0,
                atomic_seq=0.0,
                cumul_seq=0.0,
                full=0.0,
            ),

            eval_rollouts_to_log=[],
            eval_rollouts_for_videos=[],
        ),
    ),
    imsize=400, #256

    logger_config=dict(
        snapshot_gap=25,
        snapshot_mode='none',
    ),
)

env_params = {
    'pg-4obj': {
        'env_class': [PickAndPlaceEnv],
        'env_kwargs': [dict(
            # Environment dynamics
            action_scale=1.0,
            ball_radius=1.5, #1.
            boundary_dist=4,
            object_radius=1.0,
            min_grab_distance=1.0,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense", #dense_l1
            object_reward_only=False,
            # success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False, #True
            # get_image_base_render_size=(48, 48),
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
            init_position_strategy='random',

            num_objects=4,
            success_threshold=1.0,
        )],
        'rl_variant.eval_goal_sampling_mode': ['random'],

        'rl_variant.mask_variant.mask_conditioned': [False],
        # 'rl_variant.mask_variant.mask_conditioned': [True],

        'rl_variant.algo_kwargs.num_epochs': [2500],
        'rl_variant.ckpt': [
            # 'pg-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_56_41_id000--s40057',
            # 'pg-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_56_41_id000--s54673',
            # 'pg-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_56_41_id000--s57147',
            # 'pg-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_56_41_id000--s58208',
            # 'pg-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_56_41_id000--s99224',

            # 'pg-4obj/07-21-inferred-n-30-no-goal-relabeling/07-21-inferred-n-30-no-goal-relabeling_2020_07_21_20_08_52_id000--s88487',
            # 'pg-4obj/07-21-inferred-n-30-no-goal-relabeling/07-21-inferred-n-30-no-goal-relabeling_2020_07_21_20_08_58_id000--s96671',
            # 'pg-4obj/07-21-inferred-n-30-no-goal-relabeling/07-21-inferred-n-30-no-goal-relabeling_2020_07_21_20_09_01_id000--s7957',
            # 'pg-4obj/07-21-inferred-n-30-no-goal-relabeling/07-21-inferred-n-30-no-goal-relabeling_2020_07_21_20_09_01_id000--s18819',
            # 'pg-4obj/07-21-inferred-n-30-no-goal-relabeling/07-21-inferred-n-30-no-goal-relabeling_2020_07_21_20_09_01_id000--s26471',

            # 'pg-4obj/07-21-inferred-n-30-no-mask-relabeling/07-21-inferred-n-30-no-mask-relabeling_2020_07_21_12_44_56_id000--s40880',
            # 'pg-4obj/07-21-inferred-n-30-no-mask-relabeling/07-21-inferred-n-30-no-mask-relabeling_2020_07_21_12_45_02_id000--s44330',
            # 'pg-4obj/07-21-inferred-n-30-no-mask-relabeling/07-21-inferred-n-30-no-mask-relabeling_2020_07_21_12_45_06_id000--s84870',
            # 'pg-4obj/07-21-inferred-n-30-no-mask-relabeling/07-21-inferred-n-30-no-mask-relabeling_2020_07_21_12_45_18_id000--s24778',
            # 'pg-4obj/07-21-inferred-n-30-no-mask-relabeling/07-21-inferred-n-30-no-mask-relabeling_2020_07_21_13_02_10_id000--s29398',

            'pg-4obj/07-21-point/07-21-point_2020_07_21_07_57_10_id000--s10234',
            'pg-4obj/07-21-point/07-21-point_2020_07_21_07_57_10_id000--s46866',
            'pg-4obj/07-21-point/07-21-point_2020_07_21_07_57_10_id000--s50307',
            'pg-4obj/07-21-point/07-21-point_2020_07_21_07_57_10_id000--s51011',
            'pg-4obj/07-21-point/07-21-point_2020_07_21_07_57_10_id000--s75857',
        ],
        # 'rl_variant.ckpt_epoch': [
        #     1000,
        #     # 100,
        #     # None,
        # ],
        'rl_variant.mask_variant.eval_mask_distr.atomic_seq': [1.0],
    },
    'pg-4obj-maskgen': {
        'env_class': [PickAndPlaceEnv],
        'env_kwargs': [dict(
            # Environment dynamics
            action_scale=1.0,
            ball_radius=1.5,  # 1.
            boundary_dist=4,
            object_radius=1.0,
            min_grab_distance=1.0,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense",  # dense_l1
            object_reward_only=False,
            # success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,  # True
            # get_image_base_render_size=(48, 48),
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
            init_position_strategy='random',

            num_objects=4,
            success_threshold=1.0,
        )],
        'rl_variant.eval_goal_sampling_mode': ['random'],

        'rl_variant.mask_variant.mask_conditioned': [True],

        'rl_variant.algo_kwargs.num_epochs': [4000],
        'rl_variant.ckpt': [
            # 'pg-4obj-maskgen/07-25-expl-atomic-train-atomic-and-pairs/07-25-expl-atomic-train-atomic-and-pairs_2020_07_26_00_34_59_id000--s12685',
            # 'pg-4obj-maskgen/07-25-expl-atomic-train-atomic-and-pairs/07-25-expl-atomic-train-atomic-and-pairs_2020_07_26_00_34_59_id000--s18795',
            # 'pg-4obj-maskgen/07-25-expl-atomic-train-atomic-and-pairs/07-25-expl-atomic-train-atomic-and-pairs_2020_07_26_00_34_59_id000--s45112',
            # 'pg-4obj-maskgen/07-25-expl-atomic-train-atomic-and-pairs/07-25-expl-atomic-train-atomic-and-pairs_2020_07_26_00_34_59_id000--s56255',
            # 'pg-4obj-maskgen/07-25-expl-atomic-train-atomic-and-pairs/07-25-expl-atomic-train-atomic-and-pairs_2020_07_26_00_34_59_id000--s85779',

            'pg-4obj-maskgen/07-27-expl-atomic-and-pairs-train-atomic-and-pairs/07-27-expl-atomic-and-pairs-train-atomic-and-pairs_2020_07_27_08_20_08_id000--s30648',
            'pg-4obj-maskgen/07-27-expl-atomic-and-pairs-train-atomic-and-pairs/07-27-expl-atomic-and-pairs-train-atomic-and-pairs_2020_07_27_08_26_57_id000--s3509',
            'pg-4obj-maskgen/07-27-expl-atomic-and-pairs-train-atomic-and-pairs/07-27-expl-atomic-and-pairs-train-atomic-and-pairs_2020_07_27_08_27_01_id000--s3314',
            'pg-4obj-maskgen/07-27-expl-atomic-and-pairs-train-atomic-and-pairs/07-27-expl-atomic-and-pairs-train-atomic-and-pairs_2020_07_27_08_32_02_id000--s566',
            'pg-4obj-maskgen/07-27-expl-atomic-and-pairs-train-atomic-and-pairs/07-27-expl-atomic-and-pairs-train-atomic-and-pairs_2020_07_27_08_32_02_id000--s62234',
        ],
        # 'rl_variant.ckpt_epoch': [
        #     3000,
        #     # 100,
        #     # None,
        # ],
        'rl_variant.mask_variant.eval_mask_distr.atomic': [1.0],
    },
    'pb-4obj-rel': {
        'env_class': [SawyerLiftEnvGC],
        'env_kwargs': [{
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
            'random_init_bowl_pos': True,
            'bowl_type': 'heavy',
            'num_obj': 4,
            'obj_success_threshold': 0.10,

            'objs_to_reset_outside_bowl': [0],
        }],

        # ### GCRL with oracle ###
        # 'rl_variant.mask_variant.mask_conditioned': [False],
        # 'rl_variant.eval_goal_sampling_mode': ['first_obj_in_bowl_oracle'],
        #
        # ### For the vanilla RL version ###
        # 'rl_variant.contextual_mdp': [False],

        ### Disco RL ###
        'rl_variant.mask_variant.mask_conditioned': [True],
        'rl_variant.mask_variant.param_variant.mask_format': ['distribution'],
        'rl_variant.eval_goal_sampling_mode': ['obj_in_bowl'],

        'rl_variant.algo_kwargs.num_epochs': [3000],
        'rl_variant.ckpt': [
            # 'pb-4obj-rel/07-18-distr-inferred-n-30/07-18-distr-inferred-n-30_2020_07_19_06_57_28_id000--s14736',
            # 'pb-4obj-rel/07-18-distr-inferred-n-30/07-18-distr-inferred-n-30_2020_07_19_06_57_28_id000--s62041',
            # 'pb-4obj-rel/07-18-distr-inferred-n-30/07-18-distr-inferred-n-30_2020_07_19_06_57_28_id000--s72900',
            # 'pb-4obj-rel/07-18-distr-inferred-n-30/07-18-distr-inferred-n-30_2020_07_19_06_57_29_id000--s58063',
            # 'pb-4obj-rel/07-18-distr-inferred-n-30/07-18-distr-inferred-n-30_2020_07_19_06_57_29_id000--s68725',

            # 'pb-4obj-rel/07-18-distr-hard-coded/07-18-distr-hard-coded_2020_07_19_06_59_35_id000--s51878',
            # 'pb-4obj-rel/07-18-distr-hard-coded/07-18-distr-hard-coded_2020_07_19_06_59_35_id000--s77701',
            # 'pb-4obj-rel/07-18-distr-hard-coded/07-18-distr-hard-coded_2020_07_19_06_59_35_id000--s83677',
            # 'pb-4obj-rel/07-18-distr-hard-coded/07-18-distr-hard-coded_2020_07_19_06_59_35_id000--s99383',
            # 'pb-4obj-rel/07-18-distr-hard-coded/07-18-distr-hard-coded_2020_07_19_06_59_37_id000--s34153',

            'pb-4obj-rel/07-27-disco-no-relabeling/07-27-disco-no-relabeling_2020_07_27_08_12_59_id000--s60261',
            'pb-4obj-rel/07-27-disco-no-relabeling/07-27-disco-no-relabeling_2020_07_27_08_16_06_id000--s30704',
            'pb-4obj-rel/07-27-disco-no-relabeling/07-27-disco-no-relabeling_2020_07_27_08_16_07_id000--s21024',
            'pb-4obj-rel/07-27-disco-no-relabeling/07-27-disco-no-relabeling_2020_07_27_08_16_07_id000--s57337',
            'pb-4obj-rel/07-27-disco-no-relabeling/07-27-disco-no-relabeling_2020_07_27_08_16_08_id000--s754',

            # 'pb-4obj-rel/07-27-regular-rl/07-27-regular-rl_2020_07_27_08_14_35_id000--s99034',
            # 'pb-4obj-rel/07-27-regular-rl/07-27-regular-rl_2020_07_27_08_14_36_id000--s59972',
            # 'pb-4obj-rel/07-27-regular-rl/07-27-regular-rl_2020_07_27_08_14_37_id000--s24033',
            # 'pb-4obj-rel/07-27-regular-rl/07-27-regular-rl_2020_07_27_08_14_37_id000--s54305',
            # 'pb-4obj-rel/07-27-regular-rl/07-27-regular-rl_2020_07_27_08_19_21_id000--s72661',

            # 'pb-4obj-rel/07-19-point/07-19-point_2020_07_19_07_07_47_id000--s15273',
            # 'pb-4obj-rel/07-19-point/07-19-point_2020_07_19_07_07_47_id000--s24524',
            # 'pb-4obj-rel/07-19-point/07-19-point_2020_07_19_07_07_47_id000--s48038',
            # 'pb-4obj-rel/07-19-point/07-19-point_2020_07_19_07_07_47_id000--s81190',
            # 'pb-4obj-rel/07-19-point/07-19-point_2020_07_19_07_07_47_id000--s99824',
        ],
        # 'rl_variant.ckpt_epoch': [
        #     1500,
        #     # 100,
        #     # None,
        # ],
        'rl_variant.mask_variant.eval_mask_distr.atomic': [1.0],
    },
    'pb-4obj': {
        'env_class': [SawyerLiftEnvGC],
        'env_kwargs': [{
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
            'num_obj': 4,
            'obj_success_threshold': 0.10,

            # 'objs_to_reset_outside_bowl': [0, 1, 2, 3], # for goal_sampling_mode=obj_in_bowl
            'objs_to_reset_outside_bowl': [], # for goal_sampling_mode=gorund, or ground_away_from_curr_state
        }],
        'rl_variant.eval_goal_sampling_mode': [
            # 'obj_in_bowl',
            'ground_away_from_curr_state',
        ],

        'rl_variant.max_path_length': [400],

        # 'rl_variant.mask_variant.mask_conditioned': [False],
        'rl_variant.mask_variant.mask_conditioned': [True],

        'rl_variant.algo_kwargs.num_epochs': [4000],
        'rl_variant.algo_kwargs.eval_epoch_freq': [100],
        'rl_variant.ckpt': [
            # 'pb-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_46_02_id000--s13680',
            # 'pb-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_46_02_id000--s30933',
            # 'pb-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_46_02_id000--s35977',
            # 'pb-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_46_02_id000--s59054',
            # 'pb-4obj/07-21-distr-use-proper-mean-inferred-n-30/07-21-distr-use-proper-mean-inferred-n-30_2020_07_21_07_46_02_id000--s76689',

            # 'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling_2020_07_24_19_59_30_id000--s13809',
            # 'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling_2020_07_24_19_59_33_id000--s30928',
            # 'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling_2020_07_24_19_59_34_id000--s26983',
            # 'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling_2020_07_24_19_59_34_id000--s56342',
            # 'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-goal-relabeling_2020_07_24_19_59_36_id000--s99625',

            'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling_2020_07_24_20_01_45_id000--s12717',
            'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling_2020_07_24_20_01_45_id000--s33575',
            'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling_2020_07_24_20_01_49_id000--s50361',
            'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling_2020_07_24_20_01_50_id000--s496',
            'pb-4obj/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling/07-24-distr-use-proper-mean-inferred-n-30-no-mask-relabeling_2020_07_24_20_01_52_id000--s11119',

            # 'pb-4obj/07-19-point/07-19-point_2020_07_19_07_08_36_id000--s34433',
            # 'pb-4obj/07-19-point/07-19-point_2020_07_19_07_08_38_id000--s40777',
            # 'pb-4obj/07-19-point/07-19-point_2020_07_19_07_08_39_id000--s68457',
            # 'pb-4obj/07-19-point/07-19-point_2020_07_19_07_08_40_id000--s46274',
            # 'pb-4obj/07-19-point/07-19-point_2020_07_19_07_08_40_id000--s73376',
        ],
        # 'rl_variant.ckpt_epoch': [
        #     3000,
        # ],
        'rl_variant.mask_variant.eval_mask_distr.atomic_seq': [1.0],
    },
    'shelf-4obj': {
        'env_class': [FurnitureMultiworld],
        'env_kwargs': [dict(
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

            # task_type='connect',
            control_degrees='3dpos+select+connect',
            obj_joint_type='slide',
            connector_ob_type=None, #'dist',

            move_speed=0.05,
            reward_type='state_distance',
            clip_action_on_collision=True,
            light_logging=True,

            furniture_name='shelf_ivar_0678_4obj_bb',
            anchor_objects=['1_column'],
            goal_sampling_mode='uniform',
            task_type='select2',
        ),],
        'rl_variant.eval_goal_sampling_mode': ['assembled_random'],
        'rl_variant.keep_env_infos': [True],

        'rl_variant.mask_variant.mask_conditioned': [False],
        # 'rl_variant.mask_variant.mask_conditioned': [True],

        'rl_variant.algo_kwargs.num_epochs': [4000],
        'rl_variant.ckpt': [
            # 'shelf-4obj/07-26-select2-three-subtasks/07-26-select2-three-subtasks_2020_07_26_08_50_00_id000--s94704',
            # 'shelf-4obj/07-26-select2-three-subtasks/07-26-select2-three-subtasks_2020_07_26_08_50_03_id000--s5323',
            # 'shelf-4obj/07-26-select2-three-subtasks/07-26-select2-three-subtasks_2020_07_26_08_50_21_id000--s30739',
            # 'shelf-4obj/07-26-select2-three-subtasks/07-26-select2-three-subtasks_2020_07_26_08_50_22_id000--s12924',
            # 'shelf-4obj/07-26-select2-three-subtasks/07-26-select2-three-subtasks_2020_07_26_08_50_22_id000--s24008',

            'shelf-4obj/07-26-select2-gcrl/07-26-select2-gcrl_2020_07_26_08_50_22_id000--s42843',
            'shelf-4obj/07-26-select2-gcrl/07-26-select2-gcrl_2020_07_26_08_50_24_id000--s46544',
            'shelf-4obj/07-26-select2-gcrl/07-26-select2-gcrl_2020_07_26_08_50_24_id000--s84034',
            'shelf-4obj/07-26-select2-gcrl/07-26-select2-gcrl_2020_07_26_08_50_28_id000--s21069',
            'shelf-4obj/07-26-select2-gcrl/07-26-select2-gcrl_2020_07_26_08_50_29_id000--s67458',
        ],
        # 'rl_variant.ckpt_epoch': [
        #     2500,
        #     # 100,
        #     # None,
        # ],
        'rl_variant.mask_variant.eval_mask_distr.atomic_seq': [1.0],
        # 'rl_variant.mask_variant.eval_mask_distr.atomic': [1.0],
    },
}

def process_variant(variant):
    rl_variant = variant['rl_variant']

    if 'ckpt_epoch' in rl_variant:
        rl_variant['algo_kwargs']['num_epochs'] = 3
        rl_variant['algo_kwargs']['eval_epoch_freq'] = 1

    if args.debug:
        rl_variant['num_rollouts_per_epoch'] = 2
        # rl_variant['algo_kwargs']['min_num_steps_before_training'] = 200
        rl_variant['dump_video_kwargs']['columns'] = 2
        # rl_variant['save_video_period'] = 2
        variant['imsize'] = 256

    rl_variant['algo_kwargs']['num_eval_steps_per_epoch'] = (
        rl_variant['max_path_length'] * rl_variant['num_rollouts_per_epoch']
    )

    rl_variant['renderer_kwargs']['width'] = variant['imsize']
    rl_variant['renderer_kwargs']['height'] = variant['imsize']
    if args.env in ['pb-4obj', 'pb-4obj-rel']:
        variant['env_kwargs']['img_dim'] = variant['imsize']

    if args.no_video:
        rl_variant['save_video'] = False

if __name__ == "__main__":
    args = parse_args()
    args.mem_per_exp = 3.5
    mount_blacklist = []
    if args.env not in ['pb-4obj', 'pb-4obj-rel']:
        mount_blacklist += [
            'MountLocal@/home/soroush/research/bullet-manipulation',
        ]
    if args.env not in ['shelf-4obj', 'shelf-4obj-oracle-goal']:
        mount_blacklist += [
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

