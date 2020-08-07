"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_gcrl import experiment

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_leap import SawyerPushAndReachXYEnv

if __name__ == "__main__":
    variant = dict(
        num_epochs=1001,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=200,
        batch_size=1024,

        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
        ),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-4,
            std_architecture="shared",
            # num_gaussians=1,
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

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0,
            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=False,
            use_reparam_update=True,
            compute_bc=False,
            reparam_weight=0.0,
            awr_weight=0.0,
            bc_weight=0.0,

            reward_transform_kwargs=None, # r' = r + 1
            terminal_transform_kwargs=None, # t = 0
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                # dict(
                #     path="demos/icml2020/hand/pen2_sparse.npy",
                #     obs_dict=True,
                #     is_demo=True,
                # ),
                # dict(
                #     path="demos/icml2020/hand/pen_bc5.npy",
                #     obs_dict=False,
                #     is_demo=False,
                #     train_split=0.9,
                # ),
            ],
        ),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=True,
        pretrain_policy=False,
        pretrain_rl=False,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
        env_class=SawyerPushAndReachXYEnv,
        env_kwargs=dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            goal_low=(-0.20, 0.50, -0.20, 0.50),
            goal_high=(0.20, 0.70, 0.20, 0.70),
            fix_reset=False,
            sample_realistic_goals=False,
            reward_type='hand_and_puck_distance',
            invisible_boundary_wall=True,
        ),

        observation_key="state_observation",
        desired_goal_key="state_desired_goal",
        achieved_goal_key="state_achieved_goal",
    )

    search_space = {
        'seedid': range(3),
        'num_trains_per_train_loop': [1000, 4000],
        'env_kwargs.reward_type': ['puck_distance', 'hand_and_puck_distance', ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
