import joblib
import torch.optim as optim
from gym.envs.mujoco import HalfCheetahEnv

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.pytorch_util as ptu
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy

GOOD_DDPG_POLICY_PATH = '/home/vitchyr/git/railrl/data/doodads3/01-23-ddpg' \
                        '-cheetah-sweep-2/01-23-ddpg-cheetah-sweep-2-id19' \
                        '-s62223/params.pkl'


def experiment(variant):
    data = joblib.load(GOOD_DDPG_POLICY_PATH)
    expert_policy = data['policy']

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
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=expert_policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        expert_policy.to(ptu.device)
        algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=250,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        vf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            hidden_sizes=[300, 300],
        ),
        algorithm="DDPG",
        version="DDPG-init-with-expert",
        normalize=True,
        env_class=HalfCheetahEnv,
        es_kwargs=dict(
            theta=0.5,
            max_sigma=0.3,
            min_sigma=None,
        ),
    )
    search_space = {
        'env_class': [
            # CartpoleEnv,
            # SwimmerEnv,
            HalfCheetahEnv,
            # AntEnv,
            # HopperEnv,
            # InvertedDoublePendulumEnv,
        ],
        'algo_kwargs.reward_scale': [
            1,
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
        'es_kwargs.theta': [
            1, 0.5
        ],
        'es_kwargs.max_sigma': [
            1, 0.3, 0
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(2):
            run_experiment(
                experiment,
                exp_prefix="ddpg-cheetah-explore-with-expert",
                # mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=True,
            )
