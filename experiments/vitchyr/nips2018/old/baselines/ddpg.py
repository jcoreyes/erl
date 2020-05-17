import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.envs.mujoco.sawyer_gripper_env import SawyerPushXYEnv
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.ddpg.ddpg import DDPG
from railrl.torch.networks import ConcatMlp, TanhMlpPolicy


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = MultitaskToFlatEnv(env)
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = ConcatMlp(
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
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
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
            num_epochs=300,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
        ),
        env_class=SawyerPushXYEnv,
        env_kwargs=dict(
            frame_skip=50,
            only_reward_block_to_goal=False,
        ),
        algorithm='DDPG',
        version='normal',
        normalize=True,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer-sim-push-easy-ish-check-fixed-env'

    search_space = {
        'algo_kwargs.max_path_length': [
            100,
            50,
        ],
        'exploration_type': [
            'ou',
            'epsilon',
            'gaussian',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
