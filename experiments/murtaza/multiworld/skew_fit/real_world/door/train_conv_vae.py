from multiworld.core.image_env import unormalize_image
from torch import nn
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import PiecewiseLinearSchedule
from railrl.torch.vae.conv_vae import ConvVAE
from railrl.torch.vae.vae_trainer import ConvVAETrainer
from railrl.torch.grill.launcher import generate_vae_dataset
from railrl.torch.vae.conv_vae import imsize48_default_architecture
from railrl.misc.asset_loader import load_local_or_remote_file


def experiment(variant):
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
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
        t.log_loss_under_uniform(m, uniform_dataset)
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
    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'gcp'
    exp_prefix = 'skew-fit-real-world-random-policy-data-sweep'

    use_gpu = True

    variant = dict(
        num_epochs=250,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
            skew_config=dict(
                method='inv_bernoulli_p_x',
                power=2,
            ),
            skew_dataset=True,
            priority_function_kwargs=dict(
                num_latents_to_sample=10,
                sampling_method='correct',
            ),
            use_parallel_dataloading=False,
        ),
        vae=ConvVAE,
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
            dataset_path='datasets/SawyerDoorEnv_N1000__imsize48_random_oracle_split_1.npy'
        ),
        vae_kwargs=dict(
            input_channels=3,
            imsize=48,
            decoder_distribution='bernoulli',
            architecture=imsize48_default_architecture,
        ),
        save_period=50,
        beta=5,
        representation_size=16,
        uniform_dataset_path='goals/SawyerDoorEnv_N100_imsize48goals_twin_sac.npy'
    )

    search_space = {
        'beta':[1, 2.5, 5],
        'algo_kwargs.skew_config.priority_function_kwargs.num_latents_to_sample':[1, 10],
        'algo_kwargs.skew_config.power':[-1/1000, -1/100, -1/70/ -1/50, -1/25, -1/10, -1],
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
                    zone='us-central1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p100',
                        num_gpu=1,
                    )
                ),
            )
