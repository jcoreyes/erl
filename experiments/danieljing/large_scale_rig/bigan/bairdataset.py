from __future__ import print_function
import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.gan.bigan import BiGAN
from rlkit.torch.gan.bigan_trainer import BiGANTrainer
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from rlkit.launchers.config import CELEBA_DATASET
from experiments.danieljing.large_scale_rig.gan_launcher import train_gan

if __name__ == "__main__":

    variant = dict(
        region="us-west-2",
        num_epochs=100, 
        dataset = "bair",
        generate_dataset_kwargs=dict(
            image_size = 32,
            flatten = False,
            transpose = [2, 0, 1],
            shift = 0, 
            train_batch_loader_kwargs=dict(
                batch_size=100,
                num_workers=2,
            ),
            test_batch_loader_kwargs=dict(
                batch_size=100,
                num_workers=0,
            ),
        ),
        gan_trainer_class=BiGANTrainer,
        gan_class=BiGAN,
        ngpu = 1, 
        beta = 0.5,
        lr = 0.0002,
        latent_size = 256,
        output_size = 1,
        dropout = 0.2,
        generator_threshold = 3.5,
        #nc = 3,
        #ngf = 
        #ndf = 

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
        #'dropout': [0.08, 0.1, 0.12],
        #'generator_threshold': [2, 3, 4]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(train_gan, variants, run_id=0)
