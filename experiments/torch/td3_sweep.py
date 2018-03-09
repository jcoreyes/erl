from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    HopperEnv,
    Walker2dEnv,
    # HumanoidEnv,
)

from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.td3.td3 import TD3
from rllab.envs.mujoco.humanoid_env import HumanoidEnv


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())
    es = GaussianStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
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
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
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
            num_epochs=1000,
            num_steps_per_epoch=5000,
            num_steps_per_eval=10000,
            max_path_length=1000,
            min_num_steps_before_training=10000,
            batch_size=128,
            discount=0.99,

            replay_buffer_size=int(1E6),
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        es_kwargs=dict(
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        ),
        algorithm="TD3",
        version="TD3",
        env_class=HalfCheetahEnv,
    )
    search_space = {
        'env_class': [
            # HalfCheetahEnv,
            # AntEnv,
            # HopperEnv,
            # Walker2dEnv,
            HumanoidEnv,
        ],
        'algo_kwargs.reward_scale': [0.1, 1, 10],
        # 'algo_kwargs.num_updates_per_env_step': [1, 5],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(3):
            run_experiment(
                experiment,
                exp_prefix="dev-td3-sweep",
                # exp_prefix="td3-rllab-humanoid",
                # mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=False,
            )
