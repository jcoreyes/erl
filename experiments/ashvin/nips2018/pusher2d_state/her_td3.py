import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.her_replay_buffer import SimpleHerReplayBuffer
from rlkit.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.her.her_td3 import HerTd3
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
import rlkit.misc.hyperparameter as hyp


def experiment(variant):
    env = CylinderXYPusher2DEnv(**variant['env_kwargs'])
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    es = EpsilonGreedy(
        action_space=env.action_space,
        prob_random_action=0.1,
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    goal_dim = env.goal_dim
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = SimpleHerReplayBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=500,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=100,
            num_updates_per_env_step=20,
            batch_size=100,
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=0,
        ),
        normalize=True,
        algorithm='HER-TD3',
        version='normal',
    )
    n_seeds = 3
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'her-td3-pusher2d-sweep-many-sparse'

    search_space = {
        'env_kwargs.use_hand_to_obj_reward': [False],
        'env_kwargs.use_sparse_rewards': [True],
        'replay_buffer_kwargs.num_goals_to_sample': [0, 4, 8],
        'algo_kwargs.num_updates_per_env_step': [1, 5, 10],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    #     for _ in range(n_seeds):
    #         run_experiment(
    #             experiment,
    #             exp_prefix=exp_prefix,
    #             mode=mode,
    #             variant=variant,
    #         )
    run_variants(experiment, sweeper.iterate_hyperparameters())
