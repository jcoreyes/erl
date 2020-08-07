import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerPushAndPullDoorEnv
from rlkit.launchers.experiments.murtaza.multiworld_her import her_td3_experiment
from rlkit.launchers.launcher_util import run_experiment
if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=203,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=200,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=2560,
                reward_scale=100,
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        exploration_type='ou',
        es_kwargs=dict(
            max_sigma=0.3,
        ),
        replay_buffer_class=ObsDictRelabelingBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=1,
            fraction_resampled_goals_are_env_goals=0,
        ),
        env_class=SawyerPushAndPullDoorEnv,
        algorithm="HER-TD3",
        version="normal",
        env_kwargs=dict(
        ),
        normalize=False,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        # save_video_period=200,
        # do_state_exp=True,
        # init_camera=sawyer_door_env_camera,
        # save_video=True,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'sawyer_door_push_and_pull_open_her_td3_full_state_reset'

    search_space = {
        'es_kwargs.max_sigma':[.3, .8],
        'env_kwargs.num_resets_before_door_reset':[1, int(1e6)],
        'env_kwargs.num_resets_before_hand_reset':[1, int(1e6)],
        'env_kwargs.reset_hand_with_door':[True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
