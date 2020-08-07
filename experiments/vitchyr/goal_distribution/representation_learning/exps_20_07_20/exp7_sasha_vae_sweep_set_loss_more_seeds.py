import railrl.misc.hyperparameter as hyp

from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sets.vae_launcher import train_set_vae

from multiworld.envs.pygame import PickAndPlaceEnv

if __name__ == '__main__':
    variant = dict(
        create_vae_kwargs=dict(
            latent_dim=128,
            use_fancy_architecture=True,
            encoder_cnn_kwargs=dict(
                kernel_sizes=[5, 3, 3],
                n_channels=[16, 32, 64],
                strides=[3, 2, 2],
                pool_type='none',
                hidden_activation='relu',
                normalization_type='layer',
            ),
            encoder_mlp_kwargs=dict(
                hidden_sizes=[],
            ),
            decoder_dcnn_kwargs=dict(
                kernel_sizes=[3, 3, 6],
                n_channels=[32, 16, 3],
                strides=[2, 2, 3],
                paddings=[0, 0, 0],
            ),
            decoder_mlp_kwargs=dict(
                hidden_sizes=[256, 256],
            ),
            decoder_distribution='gaussian_learned_global_scalar_variance',
            use_mlp_decoder=True,
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            vae_visualization_config=dict(
                num_recons=5,
                num_samples=20,
                # debug_period=50,
                debug_period=5,
                unnormalize_images=True,
            ),
            beta=1,
            # set_loss_weight=1,
            set_loss_weight=0,
        ),
        data_loader_kwargs=dict(
            batch_size=32,
        ),
        algo_kwargs=dict(
            num_iters=41,
            num_epochs_per_iter=100,
            # num_epochs=101,
            # num_epochs=11,
        ),
        generate_set_kwargs=dict(
            num_sets=16,
            num_samples_per_set=32,
        ),
        env_class=PickAndPlaceEnv,
        env_kwargs=dict(
            # Environment dynamics
            action_scale=1.0,
            boundary_dist=4,
            ball_radius=0.75,
            object_radius=0.50,
            cursor_visual_radius=1.,
            object_visual_radius=1.,
            min_grab_distance=0.5,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense",
            success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
            object_reward_only=False,

            init_position_strategy='random',
            num_objects=2,
        ),
        num_ungrouped_images=1024,
    )
    n_seeds = 1

    search_space = {
        'vae_trainer_kwargs.beta': [
            1,
        ],
        'vae_trainer_kwargs.set_loss_weight': [
            0, 0.0001, 0.001, 0.01, 0.1, 1,
        ],
        'env_kwargs.num_objects': [
            2, 4,
        ],
        'create_vae_kwargs.use_mlp_decoder': [
            False,
        ],
        'create_vae_kwargs.decoder_mlp_kwargs.hidden_sizes': [
            [],
        ],
        'create_vae_kwargs.decoder_distribution': [
            'gaussian_learned_global_scalar_variance',
        ],
        'create_vae_kwargs.latent_dim': [
            64,
        ],
        'create_vae_kwargs.encoder_cnn_kwargs': [
            dict(
                kernel_sizes=[5, 3, 3],
                n_channels=[16, 32, 64],
                strides=[3, 2, 2],
                paddings=[0, 0, 0],

                pool_type='none',
                hidden_activation='relu',
                normalization_type='batch',
            ),
        ],
    }
    n_seeds = 1
    # mode = 'local'
    mode = 'here_no_doodad'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'exp7-sasha-vae-sweep-set-loss-more-seeds'

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        # beta = variant['vae_trainer_kwargs']['beta']
        # slw = variant['vae_trainer_kwargs']['set_loss_weight']
        for _ in range(n_seeds):
            run_experiment(
                train_set_vae,
                num_exps_per_instance=2,
                variant=variant,
                exp_name=exp_prefix,
                mode=mode,
                # slurm_config_name='gpu_low_pri',
                use_gpu=True,
            )
