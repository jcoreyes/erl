"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy

if __name__ == "__main__":
    variant = dict(
        num_epochs=11,
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
            hidden_sizes=[256, 256, ],
            max_log_std=-1,
            min_log_std=-5,
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
            compute_bc=True,

            bc_num_pretrain_steps=10000,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0,
            policy_weight_decay=1e-4,
            bc_loss_type="mse",

            rl_weight=0.0,
            use_awr_update=True,
            use_reparam_update=True,
            reparam_weight=0.0,
            awr_weight=0.0,
            bc_weight=1.0,
        ),
        num_exps_per_instance=1,
        region='us-west-1',

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                dict(
                    path="demos/icml2020/hand/pen2.npy",
                    obs_dict=True,
                    is_demo=True,
                ),
                dict(
                    path="demos/icml2020/hand/pen_bc5.npy",
                    obs_dict=False,
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
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
    )

    search_space = {
        'env': ["pen-v0", ],
        'trainer_kwargs.bc_loss_type': ["mle"],
        'trainer_kwargs.awr_loss_type': ["mle"],
        'seedid': range(3),
        'trainer_kwargs.beta': [100],
        'trainer_kwargs.use_automatic_entropy_tuning': [False],
        'policy_kwargs.max_log_std': [0, ],
        'policy_kwargs.min_log_std': [-4, ],
        # 'trainer_kwargs.reparam_weight': [0.0, 1.0],
        # 'trainer_kwargs.awr_weight': [0.0, 1.0],
        # 'trainer_kwargs.bc_weight': [0.0, 1.0],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        trainer_kwargs = variant["trainer_kwargs"]
        if not (trainer_kwargs["reparam_weight"] == 0 and trainer_kwargs["awr_weight"] == 0 and trainer_kwargs["bc_weight"] == 0):
            variants.append(variant)

    run_variants(experiment, variants, run_id=0)
