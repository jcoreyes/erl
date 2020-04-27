import torch.nn.functional as F

import railrl.misc.hyperparameter as hyp
from railrl.launchers.experiments.disentanglement.disentangled_encoder_distance_launcher import (
    use_disentangled_encoder_distance
)
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        # env_id='Point2DEnv-Train-Half-Axis-Eval-Everything-Images-v0',
        # env_id='Point2DEnv-Train-Everything-Eval-Everything-Images-48-v0',
        env_id='Point2DEnv-Train-Everything-Eval-Everything-v1',
        qf_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        policy_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        encoder_kwargs=dict(
            hidden_sizes=[64, 64],
            hidden_activation=F.tanh,
        ),
        sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=100,
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
        reward_mode='encoder_distance',
        replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
            max_size=int(1e6),
            ob_keys_to_save=[
                'state_achieved_goal',
                'state_desired_goal',
                'state_observation',
                'encoder_achieved_goal',
                'encoder_desired_goal',
                'encoder_observation',
            ],
            goal_keys=[
                'state_desired_goal',
                'encoder_desired_goal',
            ],
        ),
        encoder_key_prefix='encoder',
        exploration_goal_sampling_mode='test',
        evaluation_goal_sampling_mode='test',
        save_video_period=10,
        disentangled_qf_kwargs=dict(
            encode_state=True,
            give_each_qf_single_goal_dim=True,
        ),
        latent_dim=4,
        save_video=True,
        save_vf_heatmap=True,
        save_video_kwargs=dict(
            imsize=16,
        ),
    )

    search_space = {
        'reward_mode': [
            'encoder_distance',
            'vectorized_encoder_distance',
            'env',
        ],
        # 'encoder_kwargs.hidden_activation': [
        #     F.tanh,
        #     F.relu,
        #     F.leaky_relu,
        # ],
        # 'latent_dim': [
        #     2,
        #     4,
        #     8,
        # ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = '{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'sanity-check-encoder-wrapped-env'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                use_disentangled_encoder_distance,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=False,
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                # time_in_mins=int(2.5*24*60),
                time_in_mins=int(5*60),
              )
