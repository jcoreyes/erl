import os.path as osp
import torch.nn.functional as F
import multiworld.envs.mujoco as mwmj
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.experiments.disentanglement.launcher import \
        disentangled_grill_her_twin_sac_experiment

from rlkit.torch.vae.conv_vae import imsize48_default_architecture

if __name__ == "__main__":
    variant = dict(
        env_id='Point2DEnv-Train-Axis-Eval-Everything-Images-v0',
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
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        td3_trainer_kwargs=dict(
            tau=1e-3,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=50,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
            max_size=int(1e6),
            ob_keys_to_save=[
                'latent_observation',
                'latent_desired_goal',
                'latent_achieved_goal',
                'state_achieved_goal',
                'state_desired_goal',
                'state_observation',
            ],
            goal_keys=['latent_desired_goal', 'state_desired_goal'],
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        achieved_goal_key='latent_achieved_goal',
        vae_exploration_goal_sampling_mode='env',
        vae_evaluation_goal_sampling_mode='env',
        base_env_exploration_goal_sampling_mode='train',
        base_env_evaluation_goal_sampling_mode='test',
        vectorized=True,
        disentangled_qf_kwargs=dict(
        ),
        vae_wrapped_env_kwargs=dict(
            norm_order=1,
            reward_params=dict(
                type='vectorized_latent_distance',
                norm_order=1,
            ),
        ),
        use_vf_to_compute_policy=True,
        use_special_q_function=True,
        latent_dim=2,
        vae_n_vae_training_kwargs=dict(
            vae_class='spatialVAE',
            vae_kwargs=dict(
                input_channels=3,
            ),
            vae_trainer_kwargs=dict(
                lr=1e-3,
                beta=0,
            ),
            vae_train_epochs=50,
            num_image_examples=30000,
            vae_architecture=imsize48_default_architecture,
        ),
        # vae_path="logs/02-25-disentangle-images-relu/02-25-disentangle-images-relu_2020_02_25_12_59_17_id000--s4248/vae.pkl",

        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            imsize=48,
        ),
    )

    search_space = {
        'disentangled_qf_kwargs.encode_state': [True],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = '{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 2
    mode = 'local'
    exp_prefix = 'disentangle-extrapolate-vectorized-3'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                disentangled_grill_her_twin_sac_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(2.5*24*60),
              )
