import os.path as osp
import multiworld.envs.mujoco as mwmj
import torch.nn.functional as F
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.experiments.disentanglement.state_launcher import \
    disentangled_her_twin_sac_experiment_v2
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

if __name__ == "__main__":
    variant = dict(
        env_id='SawyerPushNIPSEasy-v0',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        encoder_kwargs=dict(
            hidden_sizes=[400, 300],
            hidden_activation=F.tanh,
        ),
        twin_sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            # qf_lr=1e-3,
            # policy_lr=1e-3,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=512,
            num_epochs=1500,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
            max_size=int(1e6),
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        exploration_goal_sampling_mode='eval',
        evaluation_goal_sampling_mode='train',
        save_video_period=10,
        save_video=True,
        disentangled_qf_kwargs=dict(
        ),
        save_video_kwargs=dict(
            save_video_period=10,
            imsize=48,
            save_vf_heatmap=False,
            video_image_env_kwargs=dict(
                init_camera=sawyer_init_camera_zoomed_in
            ),
        ),

        latent_dim=4,
    )

    search_space = {
        'disentangled_qf_kwargs.encode_state': [True],
        # 'exploration_goal_sampling_mode': ['test'],
        # 'twin_sac_trainer_kwargs.qf_lr': [1e-4],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = '{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'disentangle-vis-again-2-with-targets-and-backprop'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                disentangled_her_twin_sac_experiment_v2,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=5,
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(2.5*24*60),
              )
