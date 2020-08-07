import rlkit.torch.pytorch_util as ptu
import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.mujoco.pusher2d import Pusher2DEnv
from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.her.her_sac import HerSac
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.data_management.her_replay_buffer import SimpleHerReplayBuffer
from rlkit.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.torch.her.her_td3 import HerTd3
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants


def experiment(variant):
    env = CylinderXYPusher2DEnv(**variant['env_kwargs'])
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    goal_dim = env.goal_dim
    qf = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = ConcatMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    replay_buffer = SimpleHerReplayBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerSac(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
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
            batch_size=128,
            max_path_length=100,
            discount=0.99,
        ),
        env_kwargs=dict(
            randomize_goals=False,  # We'll set the goal manually
            use_hand_to_obj_reward=False,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=0,
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
        normalize=True,
        algorithm='HER-SAC',
        version='normal',
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'her-sac-pusher2d-sweep-sparse'

    search_space = {
        'env_kwargs.use_sparse_rewards': [True],
        'replay_buffer_kwargs.num_goals_to_sample': [0, 4, 8],
        'algo_kwargs.num_updates_per_env_step': [1, 5, 10],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters())
