import rlkit.misc.hyperparameter as hyp
# from multiworld.envs.mujoco.cameras import (
#     sawyer_pusher_camera_upright_v2,
# )
import multiworld
import multiworld.envs.mujoco
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from rlkit.launchers.experiments.vitchyr.multiworld import her_td3_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=5000,
                max_path_length=500,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=4000,
                reward_scale=100,
                render=False,
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
        env_class=SawyerDoorEnv,
        env_kwargs=dict(),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        exploration_type='ou',
        save_video_period=100,
        do_state_exp=True,
        # init_camera=sawyer_pusher_camera_upright_v2,
        imsize=48,
        save_video=False,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
    )
    search_space = {
        'algo_kwargs.base_kwargs.max_path_length': [100],
        'env_kwargs.reward_type': [
            'angle_diff_and_hand_distance',
            'angle_difference',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'sss'
    exp_prefix = 'door-env-from-state-larger-angle-range-500epoch'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                time_in_mins=20*60,
            )
