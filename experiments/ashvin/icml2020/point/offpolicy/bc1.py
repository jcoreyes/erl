"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy

from rlkit.envs.simple.point import Point

if __name__ == "__main__":
    variant = dict(
        num_epochs=10,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=200,
        batch_size=1024,
        replay_buffer_size=int(1E6),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-4,
            std_architecture="shared",
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

            bc_num_pretrain_steps=10000,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0,
            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=False,
            use_reparam_update=False,
            compute_bc=True,
            reparam_weight=0.0,
            awr_weight=0.0,
            bc_weight=1.0,

            reward_transform_kwargs=None, # r' = r + 1
            terminal_transform_kwargs=None, # t = 0
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            demo_paths=[
                dict(
                    path="demos/icml2020/point/demos100.npy",
                    # sync_dir="ashvin/icml2020/pusher/state2/random2/run13",
                    obs_dict=False, # misleading but this arg is really "unwrap_obs_dict"
                    is_demo=True,
                    train_split=0.9,
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

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,

        save_paths=True,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",

        observation_key="state_observation",
        desired_goal_key="state_desired_goal",
        achieved_goal_key="state_achieved_goal",
    )

    search_space = {
        'seedid': range(10),
        'env_class': [Point, ],
        'env_kwargs.n': [10, ],
        # 'trainer_kwargs.beta': [0.1, 1, ],
        # 'num_trains_per_train_loop': [4000],
        # 'env_kwargs.reward_type': ['puck_distance', ],
        'policy_kwargs.min_log_std': [-6, ],
        # 'trainer_kwargs.bc_weight': [0, 1],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
