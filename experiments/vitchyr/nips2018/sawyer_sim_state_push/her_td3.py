import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer
from railrl.envs.mujoco.sawyer_gripper_env import SawyerPushEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.her.her_td3 import HerTd3
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy


def experiment(variant):
    env = SawyerPushEnv(**variant['env_kwargs'])
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    es = EpsilonGreedy(
        action_space=env.action_space,
        prob_random_action=0.1,
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    goal_dim = env.goal_dim
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
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
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=300,
            num_updates_per_env_step=1,
            batch_size=100,
            discount=0.99,
        ),
        env_kwargs=dict(
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=0,
        ),
        normalize=True,
        algorithm='HER-TD3',
        version='her',
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer-sim-push-state-new-xml-v2'

    search_space = {
        'replay_buffer_kwargs.num_goals_to_sample': [0, 4],
        'env_kwargs.randomize_goals': [False, True],
        'algo_kwargs.num_updates_per_env_step': [1, 5],
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
            )