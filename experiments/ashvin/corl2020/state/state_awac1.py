import rlkit.misc.hyperparameter as hyp
from rlkit.demos.source.contextual_mdp_path_loader import EncodingContextualPathLoader
from rlkit.launchers.experiments.ashvin.awac_grasping import (
    awac_rig_experiment,
    experiment,
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
from roboverse.envs.sawyer_rig_multiobj_v0 import SawyerRigMultiobjV0

from rlkit.torch.networks import Clamp

demo_paths_1=[dict(path='sasha/complex_obj/4dof_complex_objects_demos_0.pkl',obs_dict=True, is_demo=True,),
            dict(path='sasha/complex_obj/4dof_complex_objects_demos_1.pkl',obs_dict=True,is_demo=True,),
            dict(path='sasha/complex_obj/4dof_complex_objects_demos_2.pkl',obs_dict=True,is_demo=True,)]

demo_paths_2=[dict(path='sasha/complex_obj/4dof_complex_objects_demos_0.pkl',obs_dict=True, is_demo=True,),
            dict(path='sasha/complex_obj/4dof_complex_objects_demos_1.pkl',obs_dict=True,is_demo=True,)]

demo_paths_3=[dict(path='sasha/complex_obj/4dof_complex_objects_demos_0.pkl',obs_dict=True, is_demo=True,)]

demo_paths_4=[dict(path='sasha/complex_obj/4dof_complex_objects_demos_0.pkl',obs_dict=True, is_demo=True, data_split=0.5,)]

demo_paths_5=[dict(path='sasha/complex_obj/4dof_complex_objects_demos_0.pkl',obs_dict=True, is_demo=True, data_split=0.25,)]

demo_paths_6=[dict(path='sasha/complex_obj/4dof_complex_objects_demos_0.pkl',obs_dict=True, is_demo=True, data_split=0.01,)]


if __name__ == "__main__":
    variant = dict(
        env_class=SawyerRigMultiobjV0,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=-2,
            min_log_std=-6,
            std_architecture="values",
        ),

        qf_kwargs=dict(
            hidden_sizes=[256, 256],
            output_activation=Clamp(max=0),
        ),

        trainer_kwargs=dict(
            discount=0.95,
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
            q_num_pretrain2_steps=25000, #25000
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

            awr_min_q=True,
            awr_use_mle_for_vf=True,
        ),

        max_path_length=50,
        algo_kwargs=dict(
            batch_size=1024,
            num_epochs=101,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=4000,
        ),
        replay_buffer_kwargs=dict(
            # fraction_future_context=0.2,
            # fraction_distribution_context=0.5,
            fraction_future_context=0.0,
            fraction_distribution_context=0.0,
            max_size=int(1e5),
        ),
        demo_replay_buffer_kwargs=dict(
            fraction_future_context=0.0,
            fraction_distribution_context=0.0,
            max_size=int(1e5),
        ),
        reward_kwargs=dict(
            reward_type='wrapped_env',
            #reward_type='sparse',
            # obs_type='latent',
            # epsilon=2.0,
        ),
        imsize=48,

        observation_key='state_observation',
        desired_goal_key='no_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),
        pretrained_vae_path="demos/corl2020/data/itr_200.pkl",
        # pretrained_vae_path="/home/ashvin/data/sasha/cvqvae/vqvae/run12/id0/itr_200.pkl",
        presampled_goals_path="sasha/cvqvae/vqvae/run11/id0/multiobj_goals.pkl",
        # presampled_goals_path="/home/ashvin/data/sasha/demos/multiobj_goals.pkl",

        path_loader_class=EncodingContextualPathLoader,
        path_loader_kwargs=dict(
            recompute_reward=True,
            do_preprocess=False,
            demo_paths=[
                # dict(
                #     #path='sasha/cvqvae/vqvae/run11/id0/train.pkl',
                #     path='demos/corl2020/multiobj_datasets/demo_data/train.pkl',
                #     obs_dict=True,
                #     is_demo=True,
                #     #data_split=0.1,
                # ),
            ],
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),


        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,

        evaluation_goal_sampling_mode="presampled",
        exploration_goal_sampling_mode="presampled",

        launcher_config=dict(
            unpack_variant=True,
        ),

        train_vae_kwargs=dict(
            vae_path=None,
            representation_size=4,
            beta=10.0 / 128,
            num_epochs=501,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=False,
                save_trajectories=False,
                enviorment_dataset=False,
                use_cached=False,
                vae_dataset_specific_kwargs=dict(
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
            ),
            save_period=5,
        ),
    )

    search_space = {
        "seed": range(3),
        'path_loader_kwargs.demo_paths': [demo_paths_1],
        'trainer_kwargs.beta': [0.1, 0.3, 1.0, 3.0, ],
        'trainer_kwargs.q_num_pretrain2_steps': [100001],
        'policy_kwargs.max_log_std': [0, ],
        'trainer_kwargs.discount': [0.95, 0.96, ],
        'trainer_kwargs.clip_score': [10.0],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(awac_rig_experiment, variants, run_id=3)
