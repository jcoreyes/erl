"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.mdp_path_loader import MDPPathLoader
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
        batch_size=256,
        replay_buffer_size=int(1E6),
        layer_size=256,
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

            bc_num_pretrain_steps=10000,
            q_num_pretrain_steps=10000,
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=MDPPathLoader,
        path_loader_kwargs=dict(
            demo_path=["demos/icml2020/mujoco/ant.npy"],
            # demo_off_policy_path=[
            #     "ashvin/icml2020/hand/door/demo-bc1/run3/video_*.p",
            #     "ashvin/icml2020/hand/door/demo-bc1/run4/video_*.p",
            #     "ashvin/icml2020/hand/door/demo-bc1/run5/video_*.p",
            # ],
        ),

        logger_variant=dict(
            tensorboard=True,
        ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
    )

    search_space = {
        'env': [
            # 'half-cheetah',
            # 'inv-double-pendulum',
            # 'pendulum',
            'ant',
            # 'walker',
            # 'hopper',
            # 'humanoid',
            # 'swimmer',
        ],
        'seedid': range(3),
        'trainer_kwargs.beta': [0.1, 1, 10],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
