"""The goal of this script is to just find a good setting."""
import railrl.misc.hyperparameter as hyp
import torch.nn.functional as F
from railrl.launchers.experiments.disentanglement.rig_launcher import \
    disentangled_grill_her_twin_sac_experiment
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.conv_vae import imsize48_default_architecture

if __name__ == "__main__":
    variant = dict(
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
            # batch_size=256,
            # num_epochs=50,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=1000,
            # batch_size=4,
            # num_epochs=10,
            # num_eval_steps_per_epoch=10,
            # num_expl_steps_per_train_loop=10,
            # num_trains_per_train_loop=10,
            # min_num_steps_before_training=100,
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
        disentangled_qf_kwargs=dict(
        ),
        # vectorized=True,
        vectorized=False,
        vae_wrapped_env_kwargs=dict(
            norm_order=1,
            reward_params=dict(
                # type='vectorized_latent_distance',
                type='latent_distance',
                norm_order=1,
            ),
        ),
        latent_dim=2,
        vae_path='/home/vitchyr/res/02-26-experiments-vitchyr-disentanglement-pointmass-disentanglement-rig/02-26-experiments-vitchyr-disentanglement-pointmass-disentanglement-rig_2020_02_26_19_57_00_id000--s10424/vae.pkl',
        # vae_path='/global/scratch/vitchyr/models/02-26-experiments-vitchyr-disentanglement-pointmass-disentanglement-rig_2020_02_26_19_57_00_id000--s10424/vae.pkl',
        vae_n_vae_training_kwargs=dict(
            decoder_activation='sigmoid',
            vae_kwargs=dict(
                input_channels=3,
            ),
            vae_trainer_kwargs=dict(
                lr=1e-3,
                beta=20,
            ),
            vae_train_epochs=250,
            num_image_examples=30000,
            vae_architecture=imsize48_default_architecture,
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            imsize=48,
        ),
    )

    search_space = {
        'env_id': [
            'Point2DEnv-Train-Everything-Eval-Everything-Images-48-v0',
            # 'Point2DEnv-Train-Half-Axis-Eval-Everything-Images-v0',
        ],
        'disentangled_qf_kwargs.encode_state': [True],
        # 'vectorized': [
        #     True,
        #     False,
        # ],
        # 'disentangled_qf_kwargs.give_each_qf_single_goal_dim': [
        #     True,
        #     False,
        # ],
        'have_no_disentangled_encoder': [
            True,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = '{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 5
    # mode = 'sss'
    # exp_prefix = 'disentangled-basic-test-envs-new-vae'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                disentangled_grill_her_twin_sac_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                time_in_mins=int(2.5*24*60),
              )
