"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy

if __name__ == "__main__":
    variant = dict(
        num_epochs=501,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=1024,
        replay_buffer_size=int(1E6),
        layer_size=256,
        num_layers=2,
        algorithm="SAC AWR",
        version="normal",
        collection_mode='batch',
        sac_bc=True,
        load_demos=True,
        pretrain_rl=True,

        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256] * 4,
            max_log_std=0,
            min_log_std=-6,
            std_architecture="shared",
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256]
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            alpha=0,
            use_automatic_entropy_tuning=False,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=25000,
            policy_weight_decay=1e-4,
            weight_loss=True,
            pretraining_env_logging_period=100000,
            terminal_transform_kwargs=dict(m=1, b=0),
            do_pretrain_rollouts=True,
            train_bc_on_rl_buffer=True,
            buffer_policy_sample_actions=False,

            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,
            compute_bc=False,
            awr_use_mle_for_vf=False,
            awr_sample_actions=False,
            awr_min_q=True,
        ),
        path_loader_kwargs=dict(
            demo_paths=[
                # dict(
                #     path='demos/ant_action_noise_15.npy',
                #     obs_dict=False,
                #     is_demo=True,
                #     train_split=.9,
                # ),
                # dict(
                #     path='demos/ant_off_policy_15_demos_100.npy',
                #     obs_dict=False,
                #     is_demo=False,
                # ),
            ],
        ),
        path_loader_class=DictToMDPPathLoader,
        weight_update_period=10000,

        use_validation_buffer=True,
        add_env_demos=True,
        add_env_offpolicy_data=True,
    )

    search_space = {
        'trainer_kwargs.weight_loss':[True],
        'trainer_kwargs.beta':[2, 3, 5, ],
        'train_rl':[True],
        'pretrain_rl':[True],
        'load_demos':[True],
        'pretrain_policy':[False],
        'env_id': ['HalfCheetah-v2', 'Ant-v2', 'Walker2d-v2', ],
        'seedid': range(5),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
