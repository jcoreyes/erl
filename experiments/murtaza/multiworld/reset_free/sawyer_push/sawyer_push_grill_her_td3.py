import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright, sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.images.camera import sawyer_init_camera_zoomed_in_fixed
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment

variant = dict(
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
    ),
    init_camera=sawyer_pusher_camera_upright_v2,
    grill_variant=dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=5001,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=128,
                reward_scale=1,
                render=False,
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
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
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        exploration_type='ou',
        render=False,
        training_mode='train',
        testing_mode='test',
        save_video_period=500,
        exploration_noise=.8,
        save_video=True,
    ),
    train_vae_variant=dict(
        num_epochs=5000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
        ),
        generate_vae_dataset_kwargs=dict(
            N=10000,
            oracle_dataset=True,
            use_cached=False,
            show=False,
        ),
        save_period=500,
        beta=5,
        representation_size=16,
    ),
)

search_space = {
        'env_kwargs.num_resets_before_puck_reset':[1, int(1e6)],
        'grill_variant.algo_kwargs.base_kwargs.max_path_length':[100, 500, 1000],
        'grill_variant.training_mode':['train'],
        'grill_variant.vae_path':[
            '08-08-sawyer-pusher-vae-arena-large-puck/08-08-sawyer_pusher_vae_arena_large_puck_2018_08_08_10_36_57_0000--s-8441/itr_4500.pkl',
            '08-08-sawyer-pusher-vae-arena-large-puck-beta-schedule/08-08-sawyer_pusher_vae_arena_large_puck_beta_schedule_2018_08_08_12_49_28_0000--s-93555/itr_4500.pkl',
            '08-08-sawyer-pusher-vae-arena-large-puck-beta-schedule/08-08-sawyer_pusher_vae_arena_large_puck_beta_schedule_2018_08_08_15_04_13_0000--s-86240/itr_4500.pkl'
        ]
}
sweeper = hyp.DeterministicHyperparameterSweeper(
    search_space, default_parameters=variant,
)
# n_seeds= 1
# mode='local'
# exp_prefix= 'test'

n_seeds=1
mode = 'ec2'
exp_prefix = 'sawyer_pusher_offline_vae_arena'

for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    for i in range(n_seeds):
        run_experiment(
            grill_her_td3_full_experiment,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,
            num_exps_per_instance=2,
        )
