from pathlib import Path

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.rl_launcher import disco_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='OneObject-PickAndPlace-BigBall-RandomInit-2D-v1',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        sac_trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale='auto_normalize_by_max_magnitude',
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=501,
            num_eval_steps_per_epoch=3000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        # max_path_length=2,
        # algo_kwargs=dict(
        #     batch_size=5,
        #     num_epochs=1,
        #     num_eval_steps_per_epoch=2*20,
        #     num_expl_steps_per_train_loop=2*20,
        #     num_trains_per_train_loop=10,
        #     min_num_steps_before_training=10,
        # ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.0,
            fraction_distribution_context=0.8,
            max_size=int(1e6),
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=25,
            pad_color=255,
            subpad_length=1,
            pad_length=2,
            num_columns_per_rollout=4,
            num_imgs=8,
            num_example_images_to_show=4,
            # rows=2,
            # columns=9,
        ),
        renderer_kwargs=dict(
            # create_image_format='HWC',
            # output_image_format='CWH',
            output_image_format='CHW',
            # flatten_image=True,
            # normalize_image=False,
        ),
        load_pretrained_kwargs=dict(
            filepath='20-08-18-exp13-vae-sweep-mutually-exclusive-five/20-08-18-exp13-vae-sweep-mutually-exclusive-five_2020_08_19_09_02_23_id939276--s217154/params.pkl',
            file_type='torch',
        ),
        data_loader_kwargs=dict(
            batch_size=128,
        ),
        generate_set_for_vae_pretraining_kwargs=dict(
            num_samples_per_set=128,
            set_configs=[
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        0: 3,
                        1: 3,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        0: -2,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        2: -3,
                        3: 3,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        3: -4,
                    },
                ),
                dict(
                    version='move_a_to_b',
                    offsets_from_b=(4, 0),
                    a_axis_to_b_axis={
                        0: 2,
                        1: 3,
                    },
                ),
            ],
        ),
        generate_set_for_rl_kwargs=dict(
            num_samples_per_set=128,
            set_configs=[
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        0: 3,
                        1: 3,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        0: -2,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        2: -3,
                        3: 3,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        3: -4,
                    },
                ),
                dict(
                    version='move_a_to_b',
                    offsets_from_b=(4, 0),
                    a_axis_to_b_axis={
                        0: 2,
                        1: 3,
                    },
                ),
            ],
        ),
        num_ungrouped_images=12800,
        reward_fn_kwargs=dict(
            drop_log_det_term=True,
            sqrt_reward=True,
        ),
        rig=False,
        rig_goal_setter_kwargs=dict(
            use_random_goal=True,
        ),
        use_ground_truth_reward=False,
        logger_config=dict(
            push_prefix=False,
        ),
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 1
    mode = 'sss'
    exp_name = __file__.split('/')[-1].split('.')[0].replace('_', '-') + '-more-seeds'
    print('exp_name', exp_name)

    base_dir = '/home/vitchyr/mnt/log/20-08-18-exp13-vae-sweep-mutually-exclusive-five/'
    pretrained_paths = []
    for path in Path(base_dir).rglob('params.pkl'):
        snapshot_path = path.absolute()
        if mode == 'sss':
            path = (
                    '/global/scratch/vitchyr/doodad-log-since-07-10-2020/' +
                    str(snapshot_path).split('/home/vitchyr/mnt/log/')[-1]
            )
        else:
            path = str(snapshot_path).split('/home/vitchyr/mnt/log/')[-1]
        pretrained_paths.append(path)

    search_space = {
        # 'algo_kwargs.num_epochs': [1],
        'load_pretrained_kwargs.filepath': pretrained_paths,
        'observation_key': [
            'state_observation',
        ],
        'use_ground_truth_reward': [
            False,
        ],
        'use_onehot_set_embedding': [
            False,
        ],
        'use_dummy_model': [
            False,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = list(sweeper.iterate_hyperparameters())
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(variants):
            if mode == 'local':
                variant['generate_set_for_rl_kwargs']['saved_filename'] = (
                    'manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle'
                )
                variant['algo_kwargs'] = dict(
                    batch_size=5,
                    num_epochs=1,
                    num_eval_steps_per_epoch=2*20,
                    num_expl_steps_per_train_loop=2*20,
                    num_trains_per_train_loop=10,
                    min_num_steps_before_training=10,
                )
                variant['max_path_length'] = 2
            run_experiment(
                disco_experiment,
                exp_name=exp_name,
                num_exps_per_instance=2,
                mode=mode,
                variant=variant,
                prepend_date_to_exp_name=True,
                # slurm_config_name='cpu',
                use_gpu=True,
                # gpu_id=1,
            )
