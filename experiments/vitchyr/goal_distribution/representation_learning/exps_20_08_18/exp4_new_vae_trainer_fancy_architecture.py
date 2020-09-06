import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.vae_launcher import train_set_vae

if __name__ == "__main__":
    variant = dict(
        env_id='OneObject-PickAndPlace-BigBall-RandomInit-2D-v1',
        renderer_kwargs=dict(
            output_image_format='CHW',
        ),
        create_vae_kwargs=dict(
            latent_dim=128,
            use_fancy_architecture=True,
            decoder_distribution="gaussian_fixed_unit_variance",
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            vae_visualization_config=dict(
                num_recons=10,
                num_samples=20,
                # debug_period=50,
                debug_period=20,
                unnormalize_images=True,
                image_format='CHW',
            ),
            beta=1,
            set_loss_weight=0,
        ),
        data_loader_kwargs=dict(
            batch_size=128,
        ),
        vae_algo_kwargs=dict(
            num_iters=501,
            num_epochs_per_iter=1,
            progress_csv_file_name='vae_progress.csv',
        ),
        generate_test_set_kwargs=dict(
            num_samples_per_set=128,
            # create a new test set every run
            set_configs=[
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        0: None,
                        1: None,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        0: None,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        2: None,
                        3: None,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        2: None,
                    },
                ),
            ],
        ),
        generate_train_set_kwargs=dict(
            num_sets=3,
            num_samples_per_set=128,
            saved_filename='/global/scratch/vitchyr/doodad-log-since-07-10-2020/manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle',
            # saved_filename='manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle',
        ),
        num_ungrouped_images=10000 - 3 * 128,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 2
    mode = 'sss'
    exp_prefix = 'exp4-new-vae-trainer-fancy-architecture-take2'

    search_space = {
        'vae_trainer_kwargs.beta': [
            1, 10, 50,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = list(sweeper.iterate_hyperparameters())
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(variants):
            if mode == 'local':
                variant['vae_algo_kwargs']['num_iters'] = 1
                variant['generate_train_set_kwargs']['saved_filename'] = (
                    'manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle'
                )
            run_experiment(
                train_set_vae,
                exp_name=exp_prefix,
                num_exps_per_instance=2,
                mode=mode,
                variant=variant,
                # slurm_config_name='cpu',
                use_gpu=True,
                # gpu_id=1,
            )
