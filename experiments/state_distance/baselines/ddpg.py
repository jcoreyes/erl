import random

import rlkit.misc.hyperparameter as hyp
from rlkit.envs.multitask.ant_env import GoalXYPosAnt
from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import TanhMlpPolicy, MlpQf


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = NormalizedBoxEnv(
        env,
        **variant['normalize_kwargs']
    )
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)
    es = OUStrategy(
        action_space=env.action_space,
        **variant['ou_kwargs']
    )
    obs_dim = int(env.observation_space.flat_dim)
    action_dim = int(env.action_space.flat_dim)
    obs_normalizer = TorchFixedNormalizer(obs_dim)
    action_normalizer = TorchFixedNormalizer(action_dim)
    qf = MlpQf(
        input_size=obs_dim+action_dim,
        output_size=1,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        obs_normalizer=obs_normalizer,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf,
        policy,
        exploration_policy,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
        **variant['algo_kwargs']
    )
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-ddpg-baseline-3"

    n_seeds = 1
    mode = "ec2"
    exp_prefix = "ddpg-ant-max-d-6-post-sweep"

    num_epochs = 1000
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 50

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            use_soft_update=True,
            tau=1e-3,
            batch_size=128,
            discount=0.98,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            num_updates_per_env_step=1,
        ),
        normalize_kwargs=dict(
            obs_mean=None,
            obs_std=None,
        ),
        ou_kwargs=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        version="DDPG-no-shaping",
        algorithm="DDPG",
        env_kwargs=dict(),
    )
    search_space = {
        'multitask': [True],
        'env_class': [
            # Reacher7DofXyzGoalState,
            # GoalXVelHalfCheetah,
            GoalXYPosAnt,
            # Reacher7DofXyzPosAndVelGoalState,
            # GoalXPosHopper,
            # GoalXYPosAndVelAnt,
            # GoalXPosHalfCheetah,
            # GoalXYGymPusherEnv,
            # CylinderXYPusher2DEnv,
            # GoalXPosHalfCheetah,
            # MultitaskPusher3DEnv,
            # Walker2DTargetXPos,
        ],
        # 'env_kwargs.reward_coefs': [
        #     (1, 0, 0),
        #     (0.5, 0.375, 0.125),
        # ],
        # 'env_kwargs.norm_order': [
        #     1,
        #     2,
        # ],
        # 'env_kwargs.max_speed': [
        #     0.05,
        # ],
        # 'env_kwargs.speed_weight': [
        #     None,
        # ],
        # 'env_kwargs.goal_dim_weights': [
        #     (0.1, 0.1, 0.9, 0.9),
        # ],
        # 'env_kwargs.done_threshold': [
        #     0.005, 0.001, 0.0005
        # ],
        'env_kwargs.max_distance': [
            6,
        ],
        # 'env_kwargs.action_penalty': [
        #     1e-3, 0,
        # ],
        # 'ou_kwargs.theta': [0, 0.1],
        # 'env_kwargs.min_distance': [
        #     3,
        # ],
        # 'env_kwargs.use_low_gear_ratio': [
        #     False,
        # ],
        'algo_kwargs.num_pretrain_paths': [
            0
        ],
        # 'algo_kwargs.max_path_length': [
        #     max_path_length,
        # ],
        'algo_kwargs.num_updates_per_env_step': [
            1, 5, 10
        ],
        # 'algo_kwargs.policy_pre_activation_weight': [
        #     1.,
        #     0.01,
        #     0.,
        #     0.1,
        # ],
        'algo_kwargs.reward_scale': [
            10, 100, 1000, 10000
        ],
        'algo_kwargs.discount': [
            0.98,
        ],
        'algo_kwargs.tau': [
            0.01, 0.001,
        ],
        # 'algo_kwargs.eval_with_target_policy': [
        #     True, False
        # ],
        # 'algo_kwargs.max_q_value': [
        #     0, np.inf,
        # ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
            )
