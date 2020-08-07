import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame.point2d import Point2DWallEnv
from rlkit.data_management.online_vae_log_prob_exploration_replay_buffer import \
    OnlineVaeLogProbExplorationRelabelingBuffer
from rlkit.launchers.experiments.murtaza.vae_exploration_bonus import vae_exploration_bonus_experiment
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae import vae_schedules

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=100,
            num_steps_per_eval=500,
            max_path_length=50,
            discount=0.99,
            batch_size=128,
            num_updates_per_env_step=1,
            reward_scale=1,
            vae_training_schedule=vae_schedules.every_three,
        ),
        env_class=Point2DWallEnv,
        env_kwargs=dict(
            randomize_position_on_reset=True,
            render_onscreen=False,
            ball_radius=1,
        ),
        replay_buffer_class=OnlineVaeLogProbExplorationRelabelingBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E4),
            fraction_goals_are_rollout_goals=0,
            fraction_resampled_goals_are_env_goals=0.5,
            exploration_reward_scale=1,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        render=False,
        normalize=False,
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.3,
            min_sigma=.3
        ),
        observation_key='observation',
        desired_goal_key='desired_goal',
        exploration_type='ou',
        representation_size=16,
        generate_vae_dataset_kwargs=dict(
            N=1000,
            oracle_dataset=True,
            use_cached=True,
            show=False,
        ),
        online_vae_beta=5,
        image_length=7056,
    )
    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds= 1
    # mode='local'
    # exp_prefix= 'test'

    n_seeds=1
    mode = 'ec2'
    exp_prefix = 'pointmass-online-vae-test'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                vae_exploration_bonus_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )
