import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.reset_free.sawyer_pick_and_place.sample_pick_and_place_goals import presample_goals
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v4
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnvYZ
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.launchers.experiments.murtaza.multiworld_her import her_td3_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1001,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            num_updates_per_env_step=5,
        ),
        env_class=SawyerPickAndPlaceEnvYZ,
        env_kwargs=dict(
            hide_goal_markers=True,
            hide_arm=True,
            action_scale=.02,
            reward_type="obj_distance",
            random_init=False,
        ),
        replay_buffer_class=ObsDictRelabelingBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=.5,
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
            max_sigma=.5,
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        exploration_type='ou',
        presample_goals=presample_goals,
        save_video_period=200,
        do_state_exp=True,
        init_camera=init_sawyer_camera_v4,
        save_video=True,
    )
    search_space = {
        'env_kwargs.reset_free':[True,False],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals':[0, .2],
        'algo_kwargs.min_num_steps_before_training':[128, 1000, 2000, 3000]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds= 1
    # mode='local'
    # exp_prefix= 'test'

    n_seeds=1
    mode = 'ec2'
    exp_prefix = 'sawyer_pick_and_place_reset_sweep_confirmation'

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
