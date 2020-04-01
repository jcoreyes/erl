m __future__ import print_function
import railrl.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2
from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.arglauncher import run_variants
from railrl.torch.grill.common import train_dcgan
from railrl.torch.gan.dcgan import Generator, Discriminator
from railrl.torch.gan.dcgan_trainer import DCGANTrainer
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv

if __name__ == "__main__":

    variant = dict(
        num_epochs=1000, 
        dataroot = "data/celeba",
        workers = 2, 
        batch_size = 128, 
        image_size = 64,
        dcgan_trainer_class=DCGANTrainer,
        dcgan_class=(Generator, Discriminator),
        ngpu = 1, 
        beta1 = 0.5,
        lr = 0.0002,
        nc = 3,
        nz = 100,
        ngf = 64,
        ndf = 64,
        # algo_kwargs=dict(
        #     start_skew_epoch=5000,
        #     is_auto_encoder=False,
        #     batch_size=256,
        #     lr=1e-3,
        #     skew_config=dict(
        #         method='vae_prob',
        #         power=0,
        #     ),
        #     skew_dataset=False,
        #     weight_decay=0.0,
        #     priority_function_kwargs=dict(
        #         decoder_distribution='gaussian_identity_variance',
        #         sampling_method='importance_sampling',
        #         # sampling_method='true_prior_sampling',
        #         num_latents_to_sample=10,
        #     ),
        #     use_parallel_dataloading=False,
        # ),

        save_period=25,
        logger_variant=dict(
            tensorboard=True,
        ),

        slurm_variant=dict(
            timeout_min=48 * 60,
            cpus_per_task=10,
            gpus_per_node=1,
        ),
    )
    search_space = {
        'seedid': range(1),
        'representation_size': [64]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(train_dcgan, variants, run_id=0)
