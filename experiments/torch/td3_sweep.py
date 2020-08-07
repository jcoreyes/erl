from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    HopperEnv,
    Walker2dEnv,
    # HumanoidEnv,
)

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.pytorch_util as ptu
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv,
    SawyerPushAndReachXYZEnv,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import (
    SawyerReachXYZEnv,
    SawyerXYZEnv,
    SawyerReachXYEnv)


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())
    es = GaussianStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
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
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=100,
            min_num_steps_before_training=1000,
            batch_size=128,
            discount=0.99,
            render_during_eval=True,

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
            # HumanoidEnv,
            # SawyerPushAndReachXYEnv,
            # SawyerReachXYZEnv,
            SawyerReachXYEnv,
        ],
        'algo_kwargs.discount': [0.98],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    # mode = 'local'
    mode = 'here_no_doodad'
    exp_prefix = 'dev'

    # n_seeds = 3
    # mode = 'ec2'
    exp_prefix = 'multiworld-sawyer-reacher-xy-td3-check'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
