"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.launchers.experiments.ashvin.awr_sac_rl import experiment, process_args

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
from rlkit.torch.networks import Clamp

if __name__ == "__main__":
    variant = dict(
        save_video=True,
        num_epochs=151,
        num_eval_steps_per_epoch=100,
        num_trains_per_train_loop=100,
        num_expl_steps_per_train_loop=100,
        min_num_steps_before_training=100,
        max_path_length=10,
        batch_size=1024,
        replay_buffer_size=int(1E6),

        image_env_kwargs=dict(
            imsize=48,
            init_camera=None, # the environment initializes the camera already
            transpose=True,
            normalize=True,
            recompute_reward=False,
            non_presampled_goal_img_is_garbage=True, # do not set_to_goal
        ),
        dump_video_kwargs=dict(
            save_video_period=25,
            exploration_goal_image_key="image_observation",
            evaluation_goal_image_key="image_observation",
            image_format="CWH",
        ),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
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
            compute_bc=False,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0, #25000
            policy_weight_decay=1e-4,
            q_weight_decay=0,
            bc_loss_type="mse",

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            reparam_weight=0.0,
            awr_weight=0.0,
            bc_weight=1.0,

            post_bc_pretrain_hyperparams=dict(
                bc_weight=0.0,
                compute_bc=False,
            ),

            reward_transform_kwargs=None, # r' = r + 1
            terminal_transform_kwargs=None, # t = 0
        ),

        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),


        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="observations",
            # demo_paths=[
            #     dict(
            #         path='/home/ashvin/data/sasha/spacemouse/demo_data/train.pkl',
            #         obs_dict=False,
            #         is_demo=True,
            #     ),
            # ],
        ),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,

        env="SawyerRigGrasp-v0",

    )

    search_space = {
        'seedid': range(1),
        'trainer_kwargs.bc_loss_type': ["mle"],
        'trainer_kwargs.awr_loss_type': ["mle"],
        'trainer_kwargs.beta': [0.5,], #0.2 to 0.5
        'trainer_kwargs.reparam_weight': [0.0, ],
        'trainer_kwargs.awr_weight': [1.0],
        'trainer_kwargs.bc_weight': [1.0, ],
        'policy_kwargs.std_architecture': ["values", ],
        'trainer_kwargs.clip_score': [2, ],

        # 'trainer_kwargs.compute_bc': [True, ],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.awr_min_q': [True, ],

        'trainer_kwargs.q_weight_decay': [0, ],

        'trainer_kwargs.reward_transform_kwargs': [None, ],
        'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0), ],
        'qf_kwargs.output_activation': [Clamp(max=0)],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)
    run_variants(experiment, variants, process_args, run_id=50)