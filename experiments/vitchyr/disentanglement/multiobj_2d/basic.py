import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.experiments.disentanglement.state_launcher import (
    her_sac_experiment,
)

if __name__ == "__main__":
    variant = dict(
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        twin_sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=500,
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
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            rows=2,
            columns=5,
            subpad_length=1,
            subpad_color=127,
            pad_length=1,
            pad_color=0,
        ),
        exploration_goal_sampling_mode='random',
        evaluation_goal_sampling_mode='random',
        exploration_policy_kwargs=dict(
            exploration_version='occasionally_repeat',
        ),
    )

    search_space = {
        'env_id': [
            'TwoObjectPickAndPlace2DEnv-v0',
        ],
        'save_video': [
            True,
            False,
        ],
        'exploration_policy_kwargs.exploration_version': [
            'identity',
            'occasionally_repeat',
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

    n_seeds = 5
    mode = 'htp'
    exp_name = 'pnp-fixed-init-expl-v-toggle-launch-from-basic-take3'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                her_sac_experiment,
                exp_name=exp_name,
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
                time_in_mins=int(2.5*24*60),
              )
