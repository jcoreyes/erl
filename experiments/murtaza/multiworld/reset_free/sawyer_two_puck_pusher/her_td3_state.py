import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_two_pucks import SawyerPushAndReachXYZDoublePuckEnv
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.launchers.experiments.murtaza.multiworld_her import her_td3_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1001,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=128,
                reward_scale=100,
                render=False,
                collection_mode='online-parallel',
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
        env_class=SawyerPushAndReachXYZDoublePuckEnv,
        env_kwargs=dict(
            always_start_on_same_side=True,
            goal_always_on_same_side=False,
        ),
        replay_buffer_class=ObsDictRelabelingBuffer,
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
        normalize=False,
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        exploration_type='ou',
        save_video_period=250,
        do_state_exp=True,
        init_camera=sawyer_pusher_camera_upright_v2,
        save_video=True,
        imsize=48,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
    )
    search_space = {
        'env_kwargs.reward_type':['puck1_distance', 'puck2_distance', 'state_distance'],
        'env_kwargs.num_resets_before_puck_reset':[1, int(1e6)],
        'env_kwargs.num_resets_before_hand_reset':[1, int(1e6)],
        'algo_kwargs.base_kwargs.max_path_length':[100, 500],
        'env_kwargs.always_start_on_same_side':[True, False]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds= 1
    # mode='local'
    # exp_prefix= 'test'

    n_seeds=1
    mode = 'ec2'
    exp_prefix = 'sawyer_two_puck_push_env_reset'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['env_kwargs']['num_resets_before_hand_reset'] == int(1e6)  and variant['env_kwargs']['num_resets_before_hand_reset'] == 1:
            continue
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
            )
