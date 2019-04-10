from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    HopperEnv,
    Walker2dEnv,
)
from gym.envs.mujoco import InvertedDoublePendulumEnv

from railrl.envs.pygame.point2d import Point2DEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import TwinSAC
from railrl.torch.td3.td3 import TD3


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())
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
    vf = FlattenMlp(input_size=obs_dim, output_size=1, **variant['vf_kwargs'])
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = TwinSAC(
        env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=200,
            num_steps_per_epoch=5000,
            num_steps_per_eval=10000,
            max_path_length=1000,
            min_num_steps_before_training=10000,
            # num_epochs=200,
            # num_steps_per_epoch=500,
            # num_steps_per_eval=1000,
            # max_path_length=100,
            batch_size=100,
            discount=0.99,

            replay_buffer_size=int(1E6),

            soft_target_tau=1e-3,
            policy_and_target_update_period=1,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        es_kwargs=dict(
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        ),
        algorithm="Twin-SAC",
        version="Twin-SAC-on-Q-no-delay",
        env_class=HalfCheetahEnv,
    )
    search_space = {
        'env_class': [
            HalfCheetahEnv,
            AntEnv,
            HopperEnv,
            Walker2dEnv,
            # Point2DEnv,
            # InvertedDoublePendulumEnv,
        ],
        'algo_kwargs.reward_scale': [0.01, 1, 100, 10000],
        'algo_kwargs.num_updates_per_env_step': [1],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(2):
            run_experiment(
                experiment,
                # exp_prefix="dev-twin-sace-sweep",
                exp_prefix="twin-sac-on-q-sweep",
                mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=False,
            )
