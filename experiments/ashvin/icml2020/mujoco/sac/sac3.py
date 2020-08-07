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
        num_epochs=500,
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
        qf_kwargs=dict(hidden_sizes=[256, 256]),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=True,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=100000,
            policy_weight_decay=1e-4,
            weight_loss=True,
            pretraining_env_logging_period=100000,
            terminal_transform_kwargs=dict(m=1, b=0),
            do_pretrain_rollouts=True,
            train_bc_on_rl_buffer=True,
            buffer_policy_sample_actions=False,
            validation_qlearning=True,
        ),
        use_validation_buffer=True,

        policy_kwargs=dict(
            hidden_sizes=[256]*4,
            max_log_std=0,
            min_log_std=-6,
            std_architecture="shared",
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

        add_env_demos=True,
        add_env_offpolicy_data=True,
    )

    search_space = {
        'use_weights':[True],
        'policy_kwargs.hidden_sizes':[[256]*4],
        'trainer_kwargs.use_automatic_entropy_tuning':[False],
        'trainer_kwargs.alpha':[0],
        'trainer_kwargs.weight_loss':[True],
        'trainer_kwargs.bc_num_pretrain_steps':[25000],
        'trainer_kwargs.q_num_pretrain1_steps':[0],
        'trainer_kwargs.q_num_pretrain2_steps':[0],
        # 'trainer_kwargs.beta':[
        #     .01,
        #     .1,
        #     # 1,
        #     # 10,
        #     # 100,
        # ],
        'train_rl':[True],
        'pretrain_rl':[True],
        'load_demos':[True],
        'pretrain_policy':[True],
        'env': [
            'half-cheetah',
        ],
        'policy_class':[
          GaussianPolicy,
        ],
        'trainer_kwargs.awr_loss_type':[
            'mle'
        ],
        'trainer_kwargs.reparam_weight': [1.0],
        'trainer_kwargs.awr_weight': [0.0],
        'trainer_kwargs.bc_weight': [0, 1.0, ],
        'trainer_kwargs.compute_bc': [True],
        'trainer_kwargs.use_reparam_update': [True],
        'trainer_kwargs.awr_use_mle_for_vf': [False, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.awr_min_q': [True, ],
        'trainer_kwargs.q_weight_decay': [0],
        # 'trainer_kwargs.normalize_over_batch': ["step_fn"],
        'seedid': range(5),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
