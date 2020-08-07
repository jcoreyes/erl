"""
AWR + SAC from demo experiment
"""

from rlkit.launchers.experiments.awac.awac_rl import experiment

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=1024,
        replay_buffer_size=int(1E6),

        layer_size=256,
        # policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, ],
            # max_log_std=-2,
            # min_log_std=-5,
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
            use_automatic_entropy_tuning=True,
            alpha=0,
            compute_bc=False,

            use_awr_update=True,
            use_reparam_update=True,
            reparam_weight=1.0,
            awr_weight=0.0,

            # bc_num_pretrain_steps=0,
            # q_num_pretrain_steps=0,
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        logger_variant=dict(
            tensorboard=True,
        ),
        load_demos=False,
        pretrain_policy=False,
        pretrain_rl=False,
    )

    search_space = {
        'env': [
            'half-cheetah',
            'inv-double-pendulum',
            'pendulum',
            # 'ant',
            # 'walker',
            'hopper',
            # 'humanoid',
            # 'swimmer',
        ],
        # 'trainer_kwargs.bc_loss_type': ["mse"],
        # 'trainer_kwargs.awr_loss_type': ["mle"],
        'seedid': range(3),
        # 'trainer_kwargs.beta': [0.1, 1, 10],
        'trainer_kwargs.use_automatic_entropy_tuning': [False, True],
        # 'policy_kwargs.min_log_std': [-10, -8, -6, -4, -2, ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
