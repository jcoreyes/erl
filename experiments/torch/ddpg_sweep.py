import torch.optim as optim
from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    HopperEnv,
    Walker2dEnv,
)

from railrl.envs.pygame.point2d import Point2DEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from railrl.torch.ddpg.ddpg import DDPG
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())
    es = OUStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=10,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,

            save_replay_buffer=True,
            replay_buffer_size=15000,
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        es_kwargs=dict(
            min_sigma=None,  # Constant sigma
        ),
        algorithm="DDPG",
        version="DDPG",
        normalize=True,
        env_class=HalfCheetahEnv,
    )
    search_space = {
        'env_class': [
            # InvertedPendulumEnv,
            # InvertedDoublePendulumEnv,
            # HalfCheetahEnv,
            # SwimmerEnv,
            # AntEnv,
            # HopperEnv,
            # Walker2dEnv,
            Point2DEnv,
            # InvertedDoublePendulumEnv,
        ],
        'algo_kwargs.reward_scale': [
            1,
        ],
        'algo_kwargs.policy_pre_activation_weight': [
            0,
        ],
        'algo_kwargs.optimizer_class': [
            optim.Adam,
        ],
        'algo_kwargs.tau': [
            1e-2,
        ],
        'algo_kwargs.num_updates_per_env_step': [
            1,
        ],
        'qf_kwargs.layer_norm': [
            True,
        ],
        'policy_kwargs.layer_norm': [
            True,
        ],
        'es_kwargs.theta': [
            1,
        ],
        'es_kwargs.max_sigma': [
            0.1,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(1):
            run_experiment(
                experiment,
                # exp_prefix="dev-ddpg-sweep",
                exp_prefix="ddpg-point2d-short",
                # mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=False,
            )
