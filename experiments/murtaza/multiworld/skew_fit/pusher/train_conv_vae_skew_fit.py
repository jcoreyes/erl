from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from torch import nn
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
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
    uniform_dataset=generate_uniform_dataset_reacher(
       **variant['generate_uniform_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
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
        if epoch % variant['train_weight_update_period'] == 0:
            t.update_train_weights()


if __name__ == "__main__":
    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'gcp'
    exp_prefix = 'pusher_offline_mle'

    use_gpu = True

    variant = dict(
        num_epochs=1000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
            skew_config=dict(
                method='inv_bernoulli_p_x',
                power=1/100,
            ),
            skew_dataset=False,
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
            imsize=48,
            init_camera=sawyer_init_camera_zoomed_in,
            env_id='SawyerPushNIPS-v0',
            N=5000,
            test_p=.9,
            use_cached=False,
            show=False,
            n_random_steps=100,
            non_presampled_goal_img_is_garbage=True,
            dataset_path='datasets/SawyerPushNIPS-v0_N5000_sawyer_init_camera_zoomed_in_imsize48_random_oracle_split_0.npy',
        ),
        vae_kwargs=dict(
            input_channels=3,
            imsize=48,
            architecture=imsize48_default_architecture,
            decoder_distribution='bernoulli'
        ),
        generate_uniform_dataset_kwargs=dict(
            init_camera=sawyer_init_camera_zoomed_in,
            env_id='SawyerPushNIPS-v0',
            num_imgs=1000,
            use_cached_dataset=False,
            show=False,
            save_file_prefix='pusher',
            dataset_path='datasets/pusher_N1000_imsize48uniform_images_.npy',
        ),
        save_period=50,
        representation_size=4,
        beta=10 / 128,
        train_weight_update_period=1,
    )

    search_space = {
        # 'algo_kwargs.priority_function_kwargs.sampling_method':['importance_sampling', 'correct'],
        # 'algo_kwargs.skew_config.power':[1/1000, 1/500, 1/100, 1/50, 1/10],
        # 'algo_kwargs.priority_function_kwargs.num_latents_to_sample':[10],
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
                    zone='us-west1-b',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                )
            )
