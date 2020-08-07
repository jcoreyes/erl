import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_reset import SawyerPushAndReachXYEnv
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.experiments.murtaza.state_based_vae import grill_her_td3_full_experiment
from experiments.murtaza.multiworld.reset_free.pointmass.generate_state_based_vae_dataset import generate_vae_dataset
from rlkit.pythonplusplus import identity

variant = dict(
    env_class=SawyerPushAndReachXYEnv,
    env_kwargs=dict(
        reward_type='puck_distance',
    ),
    grill_variant=dict(
        algo_kwargs=dict(
            num_epochs=1001,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=50,
            discount=0.99,
            batch_size=128,
            num_updates_per_env_step=1,
            reward_scale=1,
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
            max_sigma=.3,
            min_sigma=.3
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        exploration_type='ou',
        render=False,
        training_mode='train_env_goals',
        testing_mode='test',
        save_video=False,
    ),
    train_vae_variant=dict(
        generate_vae_data_fctn=generate_vae_dataset,
        num_epochs=100,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
        ),
        vae_kwargs=dict(
            hidden_sizes=[32, 32],
            output_activation=identity,
        ),
        generate_vae_dataset_kwargs=dict(
            N=10000,
            oracle_dataset=True,
            use_cached=False,
            env_class=SawyerPushAndReachXYEnv,
            env_kwargs=dict(
                reward_type='puck_distance',
                goal_low=(-0.1, 0.5, 0.02, -0.1, 0.6),
                goal_high=(0.1, 0.7, 0.02, 0.1, 0.7),
            ),
        ),
        beta=1,
        representation_size=5,
    ),
)

search_space = {
        'grill_variant.algo_kwargs.reward_scale':[100, 1000, 10000],
        'grill_variant.replay_buffer_kwargs.fraction_goals_are_rollout_goals':[0, .5],
        'grill_variant.replay_buffer_kwargs.fraction_resampled_goals_are_env_goals':[0, .5],
        'train_vae_variant.beta':[.01, .1, 1]
}
sweeper = hyp.DeterministicHyperparameterSweeper(
    search_space, default_parameters=variant,
)
# n_seeds= 1
# mode='local'
# exp_prefix= 'test'

n_seeds=1
mode = 'ec2'
exp_prefix = 'sawyer_pusher_state_grill'

for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    for i in range(n_seeds):
        run_experiment(
            grill_her_td3_full_experiment,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,
            num_exps_per_instance=3,
        )