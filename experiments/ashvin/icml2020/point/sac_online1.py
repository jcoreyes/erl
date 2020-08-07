"""
AWR + SAC from demo experiment
"""

from rlkit.launchers.experiments.awac.awac_rl import experiment

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy

from rlkit.envs.simple.point import Point

if __name__ == "__main__":
    variant = dict(
        num_epochs=101,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=100,
        batch_size=1024,
        replay_buffer_size=int(1E6),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
            # num_gaussians=1,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, ],
        ),

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=False,
            alpha=0,
            compute_bc=False,

            use_awr_update=False,
            use_reparam_update=True,

            # bc_num_pretrain_steps=0,
            # q_num_pretrain_steps=0,
        ),

        load_demos=False,
        pretrain_policy=False,
        pretrain_rl=False,

        env_kwargs=dict(
            n=2,
        ),
    )

    search_space = {
        'env_class': [Point, ],
        'env_kwargs.n': [2, 5, 10, 25, 50, 100],
        # 'trainer_kwargs.bc_loss_type': ["mse"],
        'seedid': range(3),
        # 'trainer_kwargs.beta': [0.0001, 0.001, 0.01], # [0.1, 1, 10],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
