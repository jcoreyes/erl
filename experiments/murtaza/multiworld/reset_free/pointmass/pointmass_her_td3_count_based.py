import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame.point2d import Point2DWallEnv
from rlkit.data_management.obs_dict_count_based_replay_buffer import ObsDictCountBasedRelabelingBuffer
from rlkit.launchers.experiments.murtaza.multiworld_her import her_td3_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=50,
            discount=0.99,
            batch_size=128,
            num_updates_per_env_step=1,
            reward_scale=1,
        ),
        env_class=Point2DWallEnv,
        env_kwargs=dict(
            ball_radius=0.5,
            render_onscreen=False,
            inner_wall_max_dist=2,
            wall_shape="u",
        ),
        replay_buffer_class=ObsDictCountBasedRelabelingBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.5,
            fraction_resampled_goals_are_env_goals=0.5,
            count_based_reward_scale=0,
            hash_dim=10,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        normalize=False,
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
        ),
        observation_key='observation',
        desired_goal_key='desired_goal',
        exploration_type='ou'
    )
    search_space = {
        'env_kwargs.randomize_position_on_reset':[True, False],
        'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': [0, .5, 1],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [0, .5, 1],
        'replay_buffer_kwargs.count_based_reward_scale': [0, 1],
        'es_kwargs.max_sigma':[.3, .4, .5]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds= 1
    # mode='local'
    # exp_prefix= 'test'

    n_seeds=1
    mode = 'ec2'
    exp_prefix = 'pointmass_wall_u_count_based_exploration'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
