import rlkit.misc.hyperparameter as hyp
# from multiworld.envs.mujoco.cameras import (
#     sawyer_pusher_camera_upright_v2,
# )
import multiworld
import multiworld.envs.mujoco
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from rlkit.launchers.experiments.vitchyr.multiworld import her_td3_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1000,
                num_steps_per_epoch=1000,
                num_steps_per_eval=5000,
                max_path_length=500,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=1000,
                reward_scale=100,
                render=False,
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
        env_id='SawyerPushAndReachXYEnv-ResetFree-v0',
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.5,
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
        init_camera=sawyer_pusher_camera_upright_v2,
        imsize=48,
        save_video=True,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
    )
    search_space = {
        'env_id': [
            'SawyerPushXyzEasyEnv-v0',
            'SawyerPushAndReachXyzEasyEnv-v0',
            'SawyerPushXyzFullArenaEnv-v0',
            'SawyerPushAndReachXyzFullArenaEnv-v0',
        ],
        # 'env_kwargs.num_resets_before_puck_reset': [int(1e6)],
        # 'env_kwargs.num_resets_before_hand_reset': [20, int(1e6)],
        'algo_kwargs.base_kwargs.max_path_length': [500],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 2
    mode = 'slurm_singularity'
    exp_prefix = 'reset-free-push-xyz-state'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                time_in_mins=20*60,
            )
