import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.ashvin.pixelcnn_launcher import train_pixelcnn
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        vqvae_path="sasha/vqvaes/best_256_vae.pkl",
        num_epochs=101,
        batch_size=64,
        n_layers=15,
        trainer_kwargs=dict(
            lr=3e-4,
        ),
        model_kwargs=dict(
            n_layers=15,
        ),

        cached_dataset_path="sasha/vqvaes/pixelcnn_data.npy", # ignores data_kwargs if used
        train_data_kwargs=dict(
            path='sasha/vqvaes/gr_train_complex_obj_images.npy',
            # max_traj=100,
        ),
        test_data_kwargs=dict(
            path='sasha/vqvaes/gr_test_complex_obj_images.npy',
            # max_traj=100,
        ),

        launcher_config=dict(
            unpack_variant=True,
        ),
    )

    search_space = {
        "seed": range(3),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(train_pixelcnn, variants, run_id=3)
