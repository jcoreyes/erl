import rlkit.misc.hyperparameter as hyp
from rlkit.data_management.her_replay_buffer import RelabelingReplayBuffer
from rlkit.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEasyEnv
from rlkit.launchers.experiments.vitchyr.multitask import tdm_td3_experiment
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks.experimental import HuberLoss

if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = 'dev'

    n_seeds = 1
    mode = "ec2"
    exp_prefix = "full-state-sawyer-push-reset-free-tdm-softtau-sweep-3"

    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                reward_scale=1,
            ),
            tdm_kwargs=dict(
                max_tau=15,
                num_pretrain_paths=0,
                reward_type='env',
            ),
            td3_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        # env_class=SawyerResetFreePushEnv,
        env_class=SawyerPushAndReachXYEasyEnv,
        env_kwargs=dict(
            reward_info=dict(
                type='euclidean',
            )
        ),
        # replay_buffer_class=HerReplayBuffer,
        replay_buffer_class=RelabelingReplayBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
            structure='norm_difference',
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        qf_criterion_class=HuberLoss,
        algorithm="TDM-TD3",
    )

    search_space = {
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [1],
        'algo_kwargs.tdm_kwargs.max_tau': [15],
        'algo_kwargs.td3_kwargs.tau': [0.001, 0.01, 0.1],
        'exploration_type': [
            'epsilon',
            'ou',
            'gaussian',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                tdm_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
