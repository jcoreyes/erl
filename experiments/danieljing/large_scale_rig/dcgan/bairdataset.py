from __future__ import print_function
import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.gan.dcgan import DCGAN
from rlkit.torch.gan.dcgan_trainer import DCGANTrainer
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from rlkit.data_management.external.bair_dataset.config import BAIR_DATASET_LOCATION
from experiments.danieljing.large_scale_rig.gan_launcher import train_gan

if __name__ == "__main__":

    variant = dict(
        num_epochs=5,
        dataset = "bair",
        generate_dataset_kwargs=dict(
            image_size = 64,
            flatten = False,
            transpose = [2, 0, 1],
            shift = 0.5,
            train_batch_loader_kwargs=dict(
                batch_size=128,
                num_workers=2,
            ),
            test_batch_loader_kwargs=dict(
                batch_size=128,
                num_workers=0,
            ),
        ),
        gan_trainer_class=DCGANTrainer,
        gan_class=DCGAN,
        ngpu = 1,
        beta = 0.5,
        lr = 0.0002,
        nc = 3,
        latent_size = 100,
        ngf = 64,
        ndf = 64,

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

    run_variants(train_gan, variants, run_id=0)
