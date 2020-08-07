"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.launchers.experiments.ashvin.awr_sac_gcrl import experiment, process_args
from roboverse.envs.sawyer_rig_gr_v0 import SawyerRigGRV0Env

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy

if __name__ == "__main__":
    variant = dict(
        save_video=True,
        num_epochs=1001,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=4000,
        max_path_length=75,
        batch_size=1024,

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
            exploration_goal_image_key="decoded_goal_image",
            evaluation_goal_image_key="decoded_goal_image",
            rows=3,
            columns=6,
            image_format="CWH",
        ),

        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            ob_keys_to_save=['state_observation', 'state_achieved_goal', "state_desired_goal"],
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
        ),
        demo_replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
        ),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-4,
            std_architecture="shared",
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
            q_num_pretrain2_steps=0, #25000
            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            compute_bc=True,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=None,
        ),
        num_exps_per_instance=1,
        region='us-west-2',
        
        reward_params=dict(
            type="latent_sparse",
            epsilon=3.0,
        ),
        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
        ),
        vae_wrapped_env_kwargs=dict(
            goal_sampling_mode='presampled',
            presampled_goals_path='presampled_goals.pkl'
        ),

        vae_path="vae.pkl",
        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=dict(
            recompute_reward=True,
            demo_paths=[
                dict(
                    path="goal_reaching_demos_fixed_goal.pkl",
                    obs_dict=True,
                    is_demo=True,
                    data_split=0.1,
                ),
            ],
        ),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,

        env_class=SawyerRigGRV0Env,
        goal_sampling_mode='presampled',
        env_kwargs=dict(),
        observation_key="latent_observation",
        desired_goal_key="latent_desired_goal",
        achieved_goal_key="latent_achieved_goal",
    )

    search_space = {
        'seedid': range(1),
        'trainer_kwargs.beta': [0.8],
        'policy_kwargs.min_log_std': [-6],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args, run_id=6)