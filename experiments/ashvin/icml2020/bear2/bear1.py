"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.launchers.experiments.ashvin.bear_rl import experiment, process_args

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
from rlkit.torch.networks import Clamp
from rlkit.torch.sac.bear import BEARTrainer

if __name__ == "__main__":
    variant = dict(
        num_epochs=1001,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
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
        trainer_class=BEARTrainer,
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            target_update_method='default',

            # use_automatic_entropy_tuning=True,
            # BEAR specific params
            mode='auto',
            kernel_choice='laplacian',
            policy_update_style='0',  # 0 is min, 1 is average (for double Qs)
            mmd_sigma=20.0,  # 5, 10, 40, 50

            target_mmd_thresh=0.05,  # .1, .07, 0.01, 0.02

            # gradient penalty hparams
            with_grad_penalty_v1=False,
            with_grad_penalty_v2=False,
            grad_coefficient_policy=0.001,
            grad_coefficient_q=1E-4,
            start_epoch_grad_penalty=24000,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_update_delay=1,
            num_steps_policy_update_only=1,

            ## advantage weighting
            use_adv_weighting=False,
            pretraining_env_logging_period=10000,
            pretraining_logging_period=1000,
            do_pretrain_rollouts=False,
            num_pretrain_steps=100000,
        ),
        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

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
        add_env_demos=True,
        add_env_offpolicy_data=True,

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
    )

    search_space = {
        'env': ["pen-sparse-v0", "door-sparse-v0", "relocate-sparse-v0", ],
        # 'trainer_kwargs.bc_loss_type': ["mle"],
        # 'trainer_kwargs.awr_loss_type': ["mle"],
        'seedid': range(4),
        # 'trainer_kwargs.beta': [0.3, ],
        # 'trainer_kwargs.reparam_weight': [0.0, ],
        # 'trainer_kwargs.awr_weight': [1.0],
        # 'trainer_kwargs.bc_weight': [1.0, ],
        # 'policy_kwargs.std_architecture': ["values", ],
        # 'trainer_kwargs.clip_score': [2, ],

        # # 'trainer_kwargs.compute_bc': [True, ],
        # 'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        # 'trainer_kwargs.awr_sample_actions': [False, True],
        # 'trainer_kwargs.awr_min_q': [True, ],

        # 'trainer_kwargs.q_weight_decay': [0, ],

        # 'trainer_kwargs.reward_transform_kwargs': [None, ],
        # 'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0), ],
        # 'qf_kwargs.output_activation': [Clamp(max=0)],
        # 'trainer_kwargs.train_bc_on_rl_buffer':[True],
        # 'policy_kwargs.num_gaussians': [1, ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
