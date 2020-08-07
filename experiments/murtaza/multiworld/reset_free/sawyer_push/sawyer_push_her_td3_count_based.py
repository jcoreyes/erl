import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_reset import SawyerPushAndReachXYEnv
from rlkit.data_management.obs_dict_count_based_replay_buffer import ObsDictCountBasedRelabelingBuffer
from rlkit.images.camera import sawyer_init_camera_zoomed_in_fixed
from rlkit.launchers.experiments.murtaza.multiworld_her import her_td3_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=5001,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=100,
            num_updates_per_env_step=4,
            batch_size=128,
            discount=0.99,
            min_num_steps_before_training=128,
            reward_scale=100,
        ),
        env_class=SawyerPushAndReachXYEnv,
        env_kwargs=dict(
            reward_type='puck_distance',
            hand_low=(-0.275, 0.275, 0.02),
            hand_high=(0.275, 0.825, .02),
            puck_low=(-0.25, 0.3),
            puck_high=(0.25, 0.8),
            goal_low=(-0.25, 0.3),
            goal_high=(0.25, 0.8),
        ),
        replay_buffer_class=ObsDictCountBasedRelabelingBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=.5,
            fraction_resampled_goals_are_env_goals=.5,
            exploration_counter_params=dict(
                num_samples=10,
                hash_dim=10,
                count_based_reward_scale=0,
            )
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
            max_sigma=.8,
            min_sigma=.8
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        exploration_type='ou',
        save_video_period=500,
        do_state_exp=True,
        init_camera=sawyer_init_camera_zoomed_in_fixed,
        save_video=True,
    )
    search_space = {
        'env_kwargs.reset_free':[False, True],
        'replay_buffer_kwargs.count_based_reward_scale': [0, 1, 10],
        'algo_kwargs.reward_scale':[1, 100],
        'replay_buffer_kwargs.hash_dim':[10, 32]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds= 1
    # mode='local'
    # exp_prefix= 'test'

    n_seeds=1
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_her_td3_max_goal_space_count_based_exp'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
            )
