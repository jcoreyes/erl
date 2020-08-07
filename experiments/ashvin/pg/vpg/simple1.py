"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.ashvin.vpg_rl import experiment, process_args

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
from rlkit.torch.vpg.vpg_trainer import VPGTrainer

from rlkit.data_management.env_replay_buffer import VPGEnvReplayBuffer

if __name__ == "__main__":
    variant = dict(
        num_epochs=10001,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=200,
        batch_size=1024,

        replay_buffer_class=VPGEnvReplayBuffer,
        replay_buffer_kwargs=dict(
            max_replay_buffer_size=int(1E6),
            discount_factor=0.99,
        ),
        demo_replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
        ),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[32, 32, ],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="shared",
            # num_gaussians=1,
        ),
        vf_kwargs=dict(
            hidden_sizes=[32, 32, ],
        ),

        trainer_class=VPGTrainer,

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=3E-4,
            vf_lr=3E-4,
            vf_iters_per_step=80,
            # reward_scale=1,
            # beta=1,
            # use_automatic_entropy_tuning=False,
            # alpha=0,

            # bc_num_pretrain_steps=0,
            # q_num_pretrain1_steps=0,
            # q_num_pretrain2_steps=10000,
            # policy_weight_decay=1e-4,
            # q_weight_decay=0,

            # rl_weight=1.0,
            # use_awr_update=True,
            # use_reparam_update=False,
            # compute_bc=False,
            # reparam_weight=0.0,
            # awr_weight=1.0,
            # bc_weight=0.0,

            # reward_transform_kwargs=None, # r' = r + 1
            # terminal_transform_kwargs=None, # t = 0
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            demo_paths=[
                dict(
                    path="demos/icml2020/pusher/demos100.npy",
                    obs_dict=False, # misleading but this arg is really "unwrap_obs_dict"
                    is_demo=True,
                ),
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

        logger_config=dict(
            snapshot_mode="gap",
            snapshot_gap=1000,
        ),
        load_demos=False,
        pretrain_policy=False,
        pretrain_rl=False,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
    )

    search_space = {
        'seedid': range(3),
        'env': ['pendulum', 'inv-double-pendulum', 'half-cheetah', 'ant', 'hopper', 'walker', ],
        'trainer_kwargs.vf_iters_per_step': [100, 1000],
        # 'num_trains_per_train_loop': [4000],
        # 'env_kwargs.reward_type': ['puck_distance', ],
        # 'policy_kwargs.min_log_std': [-6, ],
        # 'trainer_kwargs.bc_weight': [0, 1],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
