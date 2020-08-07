import rlkit.misc.hyperparameter as hyp

from rlkit.data_management.her_replay_buffer import RelabelingReplayBuffer
from rlkit.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from rlkit.envs.mujoco.sawyer_reach_env import SawyerReachXYEnv
from multiworld.envs.pygame.point2d import Point2DWallEnv

from rlkit.launchers.experiments.ashvin.multiworld_exploration import her_td3_experiment

from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=100,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=50,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=128,
            ),
            td3_kwargs=dict(),
            her_kwargs=dict(
                observation_key='observation',
                desired_goal_key='desired_goal',
            ),
        ),
        # env_class=SawyerReachXYEnv,
        env_class=Point2DWallEnv,
        env_kwargs=dict(
            ball_radius=0.5,
            render_onscreen=False,
            inner_wall_max_dist=2,
            # randomize_position_on_reset=False,
        ),
        replay_buffer_class=RelabelingReplayBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=1.0,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        normalize=True,
        algorithm='HER-TD3',
        version='normal',
        exploration_type='ou',
        es_kwargs=dict(),
    )
    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'dev'

    n_seeds = 5
    # mode = 'ec2'
    # exp_prefix = 'paper-reacher-results-full-state-oracle-ish'

    search_space = {
        # 'algo_kwargs.num_updates_per_env_step': [
        #     1,
        # ],
        # 'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': [
        #     0.0,
        #     0.5,
        #     1.0,
        # ],
        # 'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [
        #     0.2,
        #     # 1.0,
        # ],
        # 'env_kwargs.reward_info.type': [
        #     # 'hand_only',
        #     # 'shaped',
        #     'euclidean',
        # ],
        # 'exploration_type': [
        #     'epsilon',
        #     'ou',
        #     'gaussian',
        # ],
        'seed_id': range(n_seeds),
        'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': [
            0.0,
            0.5,
        ],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [
            0.2,
            1.0,
        ],
        'env_kwargs.randomize_position_on_reset': [False, True],
        'env_kwargs.wall_shape': ["u"],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(her_td3_experiment, sweeper.iterate_hyperparameters(), run_id=0)
