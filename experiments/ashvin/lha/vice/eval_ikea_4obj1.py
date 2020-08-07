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

if __name__ == '__main__':
    imsize = 48
    variant = dict(
        algo_kwargs=dict(
            num_epochs=5001,
            batch_size=2048,
            num_eval_steps_per_epoch=1500,
            num_expl_steps_per_train_loop=0,
            num_trains_per_train_loop=1, #4000,
            min_num_steps_before_training=1,
            # eval_epoch_freq=1,
            eval_only=True,
            eval_epoch_freq=100,
        ),
        max_path_length=100,
        sac_trainer_kwargs=dict(
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale=100,
            discount=0.99,
        ),
        contextual_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_future_context=0.0,
            fraction_distribution_context=0.0,
            fraction_replay_buffer_context=0.0,
            # recompute_rewards=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        expl_goal_sampling_mode='assembled_random',
        eval_goal_sampling_mode='assembled_random',
        save_env_in_snapshot=False,
        save_video=False,
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
            pad_color=0,
            pad_length=0,
            subpad_length=1,
        ),
        save_video_period=50,
        renderer_kwargs=dict(
            width=imsize,
            height=imsize,
        ),
        mask_conditioned=False,
        mask_variant=dict(
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
            idx_masks=[
                {2: 2, 3: 3},
                {4: 4, 5: 5},
            ],
            eval_rollouts_to_log=['atomic', 'atomic_seq'],
            eval_rollouts_for_videos=[],
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
            goal_type='zeros',  # reset
            reset_type='var_2dpos+no_rot',  # 'var_2dpos+var_1drot', 'var_2dpos+objs_near',

            control_degrees='3dpos+select+connect',
            obj_joint_type='slide',
            connector_ob_type=None,  # 'dist',

            move_speed=0.05,

            reward_type='state_distance',

            clip_action_on_collision=True,

            light_logging=True,

            furniture_name='shelf_ivar_0678_4obj_bb',
            anchor_objects=['1_column'],
            goal_sampling_mode='uniform',
            task_type='select2',
        ),
        logger_config=dict(
            snapshot_gap=25,
            snapshot_mode='gap_and_last',
        ),
        launcher_config=dict(
            unpack_variant=True,
        ),
        reward_trainer_kwargs=dict(
            mixup_alpha=0.5,
            data_split=0.1, # use 10% = 100 examples as data
            train_split=0.3, # use 30% of data = 30 examples for train
        ),
        # example_set_path="ashvin/lha/example_set_gen/07-22-pb-abs-example-set/07-22-pb-abs-example-set_2020_07_22_18_35_52_id000--s57269/example_dataset.npy",
        # example_set_path="ashvin/lha/example_set_gen/07-22-pb-rel-example-set/07-22-pb-rel-example-set_2020_07_22_18_36_47_id000--s21183/example_dataset.npy",
        # example_set_path="ashvin/lha/example_set_gen/07-22-pg-example-set/07-22-pg-example-set_2020_07_22_18_35_29_id000--s6012/example_dataset.npy",
        example_set_path="ashvin/lha/example_set_gen/shelf4obj_example_dataset.npy",
        ckpt=None,
        ckpt_epoch=0,
        switch_every=34,
    )

    search_space = {
        'seedid': range(1),
        'ckpt': [
            (
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id0/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id1/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id2/",
            ),
            (
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id3/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id4/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id5/",
            ),
            (
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id6/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id7/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id8/",
            ),
            (
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id9/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id10/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id11/",
            ),
            (
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id12/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id13/",
                "ashvin/lha/vice/ikea-4obj-oracle1/run11/id14/",
            ),
        ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(representation_learning_with_goal_distribution_launcher, variants, run_id=0)
