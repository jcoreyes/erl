import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.point2d_wall import MultitaskPoint2dWall
from railrl.events.beta_learning import BetaLearning
from railrl.events.networks import BetaQ, TanhFlattenMlpPolicy
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import setup_logger
from railrl.torch.networks import TanhMlpPolicy


def experiment(variant):
    # env = MultitaskPoint2dWall()
    env = MultitaskPoint2DEnv()
    es = GaussianStrategy(
        action_space=env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    beta_q = BetaQ(
        env,
        False,
        hidden_sizes=[32, 32],
    )
    policy = TanhFlattenMlpPolicy(
        env,
        hidden_sizes=[32, 32],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_kwargs']
    )
    algorithm = BetaLearning(
        env,
        exploration_policy=exploration_policy,
        beta_q=beta_q,
        policy=policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=100,
            num_steps_per_eval=100,
            max_path_length=50,
            batch_size=100,
            discount=0.8,
            render=True,
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(1E4),
            num_goals_to_sample=4,
            fraction_goals_are_rollout_goals=0.5,
            resampling_strategy='truncated_geometric',
            truncated_geom_factor=0.5,

        ),
    )
    setup_logger('name-of-beta-learning-experiment', variant=variant)
    experiment(variant)
