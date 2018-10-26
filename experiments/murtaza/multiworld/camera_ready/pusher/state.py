from railrl.launchers.experiments.murtaza.multiworld import her_td3_experiment
import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from railrl.launchers.launcher_util import run_experiment
if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=505,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=4,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=4000,
                reward_scale=100,
                render=False,
                collection_mode='online-parallel',
                parallel_env_params=dict(
                    num_workers=1,
                )

            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0,
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
            max_sigma=.5,
        ),
        exploration_type='ou',
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        env_class=SawyerPushAndReachXYEnv,
        env_kwargs=dict(
            reward_type='state_distance',
            norm_order=2,
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            num_resets_before_puck_reset=int(1),
            num_resets_before_hand_reset=int(1),
        ),
        init_camera=sawyer_pusher_camera_upright_v2,
        do_state_exp=True,
        save_video=False,

    )

    search_space = {
        'es_kwargs.max_sigma':[.3, .5]
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )


    n_seeds = 1
    mode = 'local'
    exp_prefix = 'pusher_state'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'sawyer_pusher_state_final'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                snapshot_mode='gap_and_last',
                snapshot_gap=50,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=5,
            )