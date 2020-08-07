import joblib
from torch import nn
import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.fit_skew.door.generate_uniform_dataset import generate_uniform_dataset_door
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer

def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    data = joblib.load(variant['file'])
    obs = data['obs']
    size = int(data['size'])
    dataset = obs[:size, :]
    n = int(size * .9)
    train_data = dataset[:n, :]
    test_data = dataset[n:, :]
    logger.get_snapshot_dir()
    print('SIZE: ', size)
    uniform_dataset = generate_uniform_dataset_door(
        **variant['generate_uniform_dataset_kwargs']
    )
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
        t.log_loss_under_uniform(uniform_dataset)
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
    exp_prefix = 'first10K_samples_fit_skew_fixed'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'test'
    use_gpu=True

    variant = dict(
        file='/home/murtaza/research/railrl/data/local/11-15-test/11-15-test_2018_11_15_12_57_26_id000--s13644/extra_data.pkl',
        num_epochs=1000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
            skew_config=dict(
                method='inv_bernoulli_p_x',
                # method='inv_exp_elbo',
                power=4,
            ),
            skew_dataset=True,
            priority_function_kwargs=dict(
                num_latents_to_sample=20,
                sampling_method='correct',
                decode_prob='none',
            ),
            use_parallel_dataloading=False,
        ),
        vae_kwargs=dict(
            input_channels=3,
            imsize=48,
            architecture=imsize48_default_architecture,
            decoder_distribution='bernoulli'
        ),
        vae=ConvVAE,
        dump_skew_debug_plots=True,
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
        save_period=50,
        beta=2.5,
        representation_size=16,
    )

    search_space = {
        # 'algo_kwargs.skew_config.power':[4],
        'algo_kwargs.skew_dataset':[True, False]
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
                num_exps_per_instance=2,
                snapshot_mode='gap_and_last',
                snapshot_gap=100,
            )