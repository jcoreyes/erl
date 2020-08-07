from multiworld.core.image_env import unormalize_image
from torch import nn
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import ConvVAE, ConvVAEDouble
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.grill.launcher import generate_vae_dataset
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.misc.asset_loader import load_local_or_remote_file

def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = variant['generate_vae_dataset_fn'](
        variant['generate_vae_dataset_kwargs']
    )
    uniform_dataset = load_local_or_remote_file(variant['uniform_dataset_path']).item()
    uniform_dataset = unormalize_image(uniform_dataset['image_desired_goal'])
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
    m = variant['vae'](representation_size, decoder_output_activation=nn.Sigmoid(), **variant['vae_kwargs'])
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
        t.update_train_weights()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'offline-vae-real-world-random-policy-data-gaussian-decoder'

    # n_seeds = 1
    # mode = 'gcp'
    # exp_prefix = 'skew-fit-real-world-random-policy-data-sweep-v7'

    use_gpu = True

    variant = dict(
        num_epochs=1000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
            skew_config=dict(
                method='vae_prob',
                power=-1/100,
            ),
            skew_dataset=False,
            priority_function_kwargs=dict(
                num_latents_to_sample=10,
                sampling_method='true_prior_sampling',
                decoder_distribution='gaussian_identity_variance'
            ),
            use_parallel_dataloading=False,
        ),
        vae=ConvVAEDouble,
        dump_skew_debug_plots=True,
        generate_vae_dataset_fn=generate_vae_dataset,
        generate_vae_dataset_kwargs=dict(
            N=1000,
            random_and_oracle_policy_data=True,
            random_and_oracle_policy_data_split=1,
            use_cached=True,
            imsize=48,
            non_presampled_goal_img_is_garbage=True,
            vae_dataset_specific_kwargs=dict(),
            n_random_steps=1,
            show=False,
            dataset_path='datasets/SawyerDoorEnv_N1000__imsize48_random_oracle_split_1.npy',
        ),
        vae_kwargs=dict(
            input_channels=3,
            imsize=48,
            decoder_distribution='gaussian',
            architecture=imsize48_default_architecture,
        ),
        save_period=50,
        beta=5,
        representation_size=16,
        uniform_dataset_path='goals/SawyerDoorEnv_N100_imsize48goals_twin_sac.npy'
    )

    search_space = {
        'beta':[.01, .1, 1, 2.5, 5, 10],
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
                    zone='us-east4-a',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p4',
                        num_gpu=1,
                    )
                ),
                skip_wait=True,
            )
