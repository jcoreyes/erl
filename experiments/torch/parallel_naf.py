"""
Experiment with NAF.
"""
import railrl.torch.pytorch_util as ptu
from railrl.envs.remote import RemoteRolloutEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.algos.parallel_naf import ParallelNAF
from railrl.torch.naf import NafPolicy
from railrl.torch.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy
from rllab.envs.mujoco.inverted_double_pendulum_env import \
    InvertedDoublePendulumEnv
from rllab.envs.normalized_env import normalize


def example(variant):
    env_class = variant['env_class']
    env_params = {}
    # Only create an env for the obs/action spaces
    env = env_class(**env_params)
    if variant['normalize_env']:
        env = normalize(env)
    obs_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    es_class = OUStrategy
    es_params = dict(
        action_space=action_space,
    )
    policy_class = NafPolicy
    policy_params = dict(
        obs_dim=int(obs_space.flat_dim),
        action_dim=int(action_space.flat_dim),
        hidden_size=400,
    )
    es = es_class(**es_params)
    policy = policy_class(**policy_params)
    exploration_policy = UniversalPolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    remote_env = RemoteRolloutEnv(
        env,
        policy,
        exploration_policy,
        variant['max_path_length'],
        variant['normalize_env'],
    )
    algorithm = ParallelNAF(
        remote_env,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    max_path_length = 100
    variant = dict(
        algo_params=dict(
            num_epochs=50,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=1024,
            replay_buffer_size=50000,
            max_path_length=max_path_length,
        ),
        max_path_length=max_path_length,
        env_class=InvertedDoublePendulumEnv,
        parallel=True,
        normalize_env=True,
    )
    run_experiment(
        example,
        exp_prefix="parallel-naf-example",
        seed=0,
        mode='here',
        variant=variant,
        use_gpu=False,
    )
