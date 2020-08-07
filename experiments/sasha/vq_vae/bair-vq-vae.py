import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize48_default_architecture_with_more_hidden_layers
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.grill.cvae_experiments import (
    grill_her_td3_offpolicy_online_vae_full_experiment,
)
from rlkit.data_management.external.bair_dataset import bair_dataset
from rlkit.torch.grill.common import train_vae
from rlkit.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from rlkit.data_management.online_conditional_vae_replay_buffer import \
        OnlineConditionalVaeRelabelingBuffer

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        beta_schedule_kwargs=dict(
            x_values=(0, 10000),
            y_values=(1, 50),
        ),
        beta=10,
        num_epochs=1000,
        dump_skew_debug_plots=False,
        decoder_activation='sigmoid',
        use_linear_dynamics=False,
        generate_vae_data_fctn=bair_dataset.generate_dataset,
        generate_vae_dataset_kwargs=dict(
            train_batch_loader_kwargs=dict(
                batch_size=256,
                num_workers=10,
            ),
            test_batch_loader_kwargs=dict(
                batch_size=256,
                num_workers=0,
            ),
        ),
        vae_trainer_class=VQ_VAETrainer,
        vae_class=VQ_VAE,
        vae_kwargs=dict(
            input_channels=3,
            # num_hiddens=256,
            # num_residual_layers=64,
            # num_residual_hiddens=3,
            # num_embeddings=1024,
            # commitment_cost=0.25,
        ),
        algo_kwargs=dict(
            start_skew_epoch=5000,
            is_auto_encoder=False,
            batch_size=256,
            lr=1e-3,
            skew_config=dict(
                method='vae_prob',
                power=0,
            ),
            skew_dataset=False,
            weight_decay=0.0,
            priority_function_kwargs=dict(
                decoder_distribution='gaussian_identity_variance',
                sampling_method='importance_sampling',
                # sampling_method='true_prior_sampling',
                num_latents_to_sample=10,
            ),
            use_parallel_dataloading=False,
        ),

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

<<<<<<< HEAD
    run_variants(train_vae, variants, run_id=13)
=======
    run_variants(train_vae, variants, run_id=13)
>>>>>>> parent of 595cfb8... Unstage experiments
