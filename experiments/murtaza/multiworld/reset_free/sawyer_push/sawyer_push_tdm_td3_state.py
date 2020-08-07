import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v3, sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from rlkit.launchers.experiments.murtaza.multiworld_tdm import tdm_td3_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=5001,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1, #TODO: RE-RUN WITH PROPER DISCOUNT
                min_num_steps_before_training=128,
                reward_scale=100,
                render=False,
            ),
            tdm_kwargs=dict(
                max_tau=15,
            ),
            td3_kwargs=dict(
                tau=1,
            ),
        ),
        env_class=SawyerPushAndReachXYEnv,
        env_kwargs=dict(
            reward_type='puck_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            num_resets_before_puck_reset=int(1e6),
            norm_order=2,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        exploration_type='ou',
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.5,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        algorithm="TDM-TD3",
        render=False,
        save_video=True,
        normalize=False,
        es_kwargs=dict(
            max_sigma=.8,
        ),
        save_video_period=200,
        do_state_exp=True,
        init_camera=sawyer_pusher_camera_upright_v2,
    )
    search_space = {
        'env_kwargs.num_resets_before_puck_reset':[1, int(1e6)],
        'env_kwargs.reward_type': ['state_distance', 'vectorized_state_distance'],
        'algo_kwargs.tdm_kwargs.max_tau': [5, 20],
        'algo_kwargs.base_kwargs.num_updates_per_env_step':[1, 4],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds= 1
    # mode='local'
    # exp_prefix= 'test'

    n_seeds=1
    mode = 'ec2'
    exp_prefix = 'sawyer_push_env_sweep_tdm_td3_reset_vs_reset_free'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                tdm_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
            )
