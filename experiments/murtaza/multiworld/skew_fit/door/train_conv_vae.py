from torch import nn
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0

from experiments.murtaza.multiworld.skew_fit.door.generate_uniform_dataset import generate_uniform_dataset_door
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.grill.launcher import generate_vae_dataset
from rlkit.torch.vae.conv_vae import ConvVAE, imsize48_default_architecture
from rlkit.torch.vae.vae_trainer import ConvVAETrainer

def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = variant['generate_vae_dataset_fn'](
        variant['generate_vae_dataset_kwargs']
    )
    uniform_dataset=generate_uniform_dataset_door(
       **variant['generate_uniform_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        # kwargs = variant['beta_schedule_kwargs']
        # kwargs['y_values'][2] = variant['beta']
        # kwargs['x_values'][1] = variant['flat_x']
        # kwargs['x_values'][2] = variant['ramp_x'] + variant['flat_x']
        variant['beta_schedule_kwargs']['y_values'][-1] = variant['beta']
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = variant['vae'](representation_size, **variant['vae_kwargs'])
    m.to(ptu.device)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.log_loss_under_uniform(m, uniform_dataset, variant['algo_kwargs']['priority_function_kwargs'])
        t.test_epoch(epoch, save_reconstruction=should_save_imgs,
                     save_scatterplot=should_save_imgs)
        if should_save_imgs:
            t.dump_samples(epoch)
            if variant['dump_skew_debug_plots']:
                t.dump_best_reconstruction(epoch)
                t.dump_worst_reconstruction(epoch)
                t.dump_sampling_histogram(epoch)
                t.dump_uniform_imgs_and_reconstructions(dataset=uniform_dataset, epoch=epoch)
        if epoch % variant['train_weight_update_period'] == 0:
            t.update_train_weights()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 1
    # mode = 'gcp'
    # exp_prefix = 'sawyer_door_fit_skew_finalized_tests'

    use_gpu = True

    variant = dict(
        num_epochs=1000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
            skew_config=dict(
                method='skew-fit',
                power=1/100,
            ),
            skew_dataset=True,
            priority_function_kwargs=dict(
                num_latents_to_sample=20,
                sampling_method='true_prior_sampling',
                decoder_distribution='gaussian_identity_variance'
            ),
            use_parallel_dataloading=False,
        ),
        vae=ConvVAE,
        dump_skew_debug_plots=True,
        generate_vae_dataset_fn=generate_vae_dataset,
        generate_vae_dataset_kwargs=dict(
            env_id='SawyerDoorHookResetFreeEnv-v0',
            init_camera=sawyer_door_env_camera_v0,
            N=100,
            oracle_dataset=False,
            use_cached=True,
            random_and_oracle_policy_data=True,
            random_and_oracle_policy_data_split=.9,
            imsize=48,
            non_presampled_goal_img_is_garbage=True,
            vae_dataset_specific_kwargs=dict(),
            policy_file='11-09-her-twin-sac-door/11-09-her-twin-sac-door_2018_11_10_02_17_10_id000--s16215/params.pkl',
            n_random_steps=100,
            show=False,
            dataset_path='datasets/SawyerDoorHookResetFreeEnv-v0_N5000_sawyer_door_env_camera_v0_imsize48_random_oracle_split_1.npy',
        ),
        vae_kwargs=dict(
            input_channels=3,
            imsize=48,
            architecture=imsize48_default_architecture,
            decoder_distribution='gaussian_identity_variance'
        ),
        generate_uniform_dataset_kwargs=dict(
            env_id='SawyerDoorHookResetFreeEnv-v0',
            init_camera=sawyer_door_env_camera_v0,
            num_imgs=1000,
            use_cached_dataset=False,
            policy_file='11-09-her-twin-sac-door/11-09-her-twin-sac-door_2018_11_10_02_17_10_id000--s16215/params.pkl',
            show=False,
            path_length=100,
            dataset_path='datasets/SawyerDoorHookResetFreeEnv-v0_N1000_imsize48uniform_images_.npy',
        ),
        save_period=10,
        beta=2.5,
        representation_size=16,
        train_weight_update_period=1,
    )

    search_space = {
        'algo_kwargs.priority_function_kwargs.sampling_method':['importance_sampling', 'true_prior_sampling'],
        'algo_kwargs.skew_config.power':[-1/100, 0],
        'algo_kwargs.priority_function_kwargs.num_latents_to_sample':[1, 10, 20],

        # 'train_weight_update_period':[1, 2, 4, 10],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=use_gpu,
                num_exps_per_instance=1,
                snapshot_mode='gap_and_last',
                snapshot_gap=100,
                gcp_kwargs=dict(
                    zone='us-east4-a    ',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p4',
                        num_gpu=1,
                    )
                )
            )
