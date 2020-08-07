"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.encoder_path_loader import EncoderPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from multiworld.envs.rlbench.rlbench_env import RLBenchEnv
from rlbench.tasks.open_drawer import OpenDrawer

if __name__ == "__main__":
    variant = dict(
        num_epochs=101,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=50,
        batch_size=1024,
        replay_buffer_size=int(1E6),
        algorithm="SAC",
        version="normal",
        collection_mode='batch',

        layer_size=256,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
        ),

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

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=1000,
            policy_weight_decay=1e-4,
            bc_loss_type="mle",
            bc_weight=0.0,

            policy_update_period=2,
            q_update_period=1,
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=EncoderPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                dict(
                    path="demos/icml2020/rlbench/rlbench.npy",
                    obs_dict=True,
                    is_demo=True,
                ),
                dict(
                    path="demos/icml2020/rlbench/rlbench_bc1.npy",
                    obs_dict=True,
                    is_demo=False,
                    train_split=0.9,
                ),
            ],
        ),

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,

        env_class=RLBenchEnv,
        env_kwargs=dict(
            task_class=OpenDrawer,
            fixed_goal=(),
            headless=False,
            camera=(500, 300),
        ),

        model_kwargs=dict(
            decoder_distribution='gaussian_identity_variance',
            input_channels=3,
            imsize=224,
            architecture=dict(
                hidden_sizes=[200, 200],
            ),
            delta_features=True,
            pretrained_features=False,
        ),
        model_path="/home/ashvin/data/s3doodad/facebook/models/rfeatures/multitask1/run2/id2/itr_4000.pt",

        desired_trajectory="/home/ashvin/code/gitignore/rlbench/demo_door_fixed2/demos5b_10_dict.npy",

        config_params = dict(
            initial_type="use_initial_from_trajectory",
            goal_type="",
            use_initial=True,
        ),
        reward_params_type="regression_distance",
    )

    search_space = {
        'seedid': range(3),
        'trainer_kwargs.beta': [10, 100, 1000],
        'trainer_kwargs.bc_weight': [0.0, 1.0],
        'trainer_kwargs.q_num_pretrain2_steps': [1000],
        # 'deterministic_exploration': [True, False],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
