import torch.optim as optim
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, \
    InvertedDoublePendulumEnv, SwimmerEnv

from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from railrl.torch.algos.ddpg import DDPG
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())
    es = OUStrategy(action_space=env.action_space)
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
            num_epochs=301,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=64,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        algorithm="DDPG",
        version="DDPG",
        normalize=True,
        env_class=HalfCheetahEnv,
    )
    search_space = {
        'env_class': [
            InvertedPendulumEnv,
            InvertedDoublePendulumEnv,
            HalfCheetahEnv,
            SwimmerEnv,
            # AntEnv,
            # HopperEnv,
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
            True, False
        ],
        'policy_kwargs.layer_norm': [
            True, False
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(3):
            run_experiment(
                experiment,
                exp_prefix="ddpg-sweep-layer-norm",
                mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=False,
            )
