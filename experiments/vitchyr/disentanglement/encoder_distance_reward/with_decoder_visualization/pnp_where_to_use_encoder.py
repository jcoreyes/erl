import torch.nn.functional as F

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.disentanglement.contextual_encoder_distance_launcher import (
    encoder_goal_conditioned_sac_experiment
)
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='OneObjectPickAndPlace2DEnv-v0',
        qf_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        policy_kwargs=dict(
            hidden_sizes=[64, 64],
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
            num_epochs=500,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            # num_epochs=5,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=300,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.5,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        save_video=True,
        # save_video=False,
        save_video_kwargs=dict(
            save_video_period=50,
            # save_video_period=1,
            rows=1,
            columns=2,
            subpad_length=1,
            subpad_color=127,
            pad_length=1,
            pad_color=0,
        ),
        evaluation_goal_sampling_mode='random',
        exploration_goal_sampling_mode='random',
        exploration_policy_kwargs=dict(
            exploration_version='occasionally_repeat',
            repeat_prob=0.5,
        ),
        disentangled_qf_kwargs=dict(
            encode_state=True,
        ),
        policy_with_encoder_kwargs=dict(),
        encoder_kwargs=dict(
            output_size=8,
            hidden_sizes=[64, 64],
            hidden_activation=F.tanh,
        ),
        renderer_kwargs=dict(
            width=32,
            height=32,
        ),
        debug_renderer_kwargs=dict(
            width=16,
            height=16,
            sweep='goal',
        ),
    )

    search_space = {
        'use_target_encoder_for_reward': [
            False,
        ],
        'encoder_reward_scale': [
            1.,
        ],
        'encoder_kwargs.output_size': [
            8,
            # 32,
        ],
        'encoder_kwargs.hidden_sizes': [
            [64, 64],
        ],
        'encoder_kwargs.hidden_activation': [
            F.leaky_relu,
        ],
        'vectorized_reward': [
            True,
            False,
        ],
        'disentangled_qf_kwargs.architecture': [
            'splice',
            'single_head_match_many_heads',
            'many_heads',
        ],
        'disentangled_qf_kwargs.encode_state': [
            True,
            False,
        ],
        'policy_with_encoder_kwargs.encode_state': [
            True,
            False,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 1
    mode = 'ec2'
    exp_name = 'pnp-1obj-sweep-where-to-use-encoder-ec2-c5-one-seeds-take4'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for seed in range(n_seeds):
            variant['exp_id'] = exp_id
            # variant['seed'] = seed
            run_experiment(
                encoder_goal_conditioned_sac_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=False,
                num_exps_per_instance=3,
                # slurm_config_name='cpu_co',
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(2.5*24*60),
            )
