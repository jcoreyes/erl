import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.e2e_rig.launcher import pointmass_fixed_goal_experiment
from railrl.torch.vae.conv_vae import imsize48_default_architecture

if __name__ == "__main__":
    variant = dict(
        vae_latent_size=8,
        vae_kwargs=dict(
            architecture=imsize48_default_architecture,
        ),
        env_kwargs=dict(
            fixed_goal=(0, 0),
            images_are_rgb=True,
            render_onscreen=False,
            show_goal=True,
            ball_radius=2,
            render_size=8,
        ),
        e2e_trainer_kwargs=dict(
            combined_lr=1e-3,
            vae_training_method='vae',
        ),
        sac_trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            target_entropy=-1,
        ),
        algorithm_kwargs=dict(
            max_path_length=50,
            batch_size=256,
            num_epochs=50,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=500,
        ),
        cnn_kwargs=dict(
            kernel_sizes=[3],
            n_channels=[64],
            strides=[1],
            hidden_sizes=[64],
            paddings=[1],
        ),
        qf_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        policy_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        replay_buffer_size=int(5E4),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 3
    mode = 'sss'
    exp_prefix = 'e2e-sac-vae-train-method-sweep-3'

    search_space = {
        'e2e_trainer_kwargs.vae_training_method': [
            'vae',
            'none',
            'vae_and_qf'
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                pointmass_fixed_goal_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                use_gpu=True,
                time_in_mins=int(2.5*24*60),
            )
