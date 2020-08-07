import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants
from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_experiments import train_rfeatures_model
from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_trainer import TimePredictionTrainer, LatentPathPredictorTrainer, GeometricTimePredictorTrainer

if __name__ == "__main__":
    variant = dict(
        output_classes=20,
        representation_size=128,
        num_epochs=100000,
        dump_skew_debug_plots=False,
        decoder_activation='gaussian',
        batch_size=80,
        dataset_kwargs=dict(
            test_p=.9,
            dataset_name=None,
        ),
        trainer_class=TimePredictionTrainer,
        model_kwargs=dict(
            decoder_distribution='gaussian_identity_variance',
            input_channels=3,
            imsize=224,
            architecture=dict(
                hidden_sizes=[200, 200],
            ),
            delta_features=False,
            pretrained_features=False,
        ),
        trainer_kwargs=dict(
            lr=1e-3,
            loss_weights=dict(
                mse=1.0,
            )
        ),
        save_period=100,
        slurm_variant=dict(
            timeout_min=48 * 60,
            cpus_per_task=10,
            gpus_per_node=1,
        ),
        num_train_workers=8,
    )

    search_space = {
        "dataset_kwargs.dataset_name": ["multitask_v1_L100"],
        "model_kwargs.delta_features": [True, ],
        "model_kwargs.pretrained_features": [True, False, ],
        "model_kwargs.normalize": [True, ],
        "trainer_kwargs.loss_weights.mse": [0.0, 1.0],
        "trainer_kwargs.loss_weights.classification_gradients": [False, True, ],
        "trainer_class": [GeometricTimePredictorTrainer],
        "seedid": range(1),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    run_variants(train_rfeatures_model, sweeper.iterate_hyperparameters(), run_id=2)
