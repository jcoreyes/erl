import pickle

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.wrappers import convert_gym_space
from railrl.policies.simple import ZeroPolicy
from railrl.samplers.path_sampler import MultitaskPathSampler


def get_replay_buffer(variant, save_replay_buffer=False):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    if variant['generate_data']:
        action_space = convert_gym_space(env.action_space)
        es = variant['sampler_es_class'](
            action_space=action_space,
            **variant['sampler_es_params']
        )
        exploration_policy = ZeroPolicy(
            int(action_space.flat_dim),
        )
        replay_buffer_size = variant['replay_buffer_size']
        replay_buffer = SplitReplayBuffer(
            EnvReplayBuffer(
                replay_buffer_size,
                env,
                flatten=True,
            ),
            EnvReplayBuffer(
                replay_buffer_size,
                env,
                flatten=True,
            ),
            fraction_paths_in_train=0.8,
        )
        sampler = MultitaskPathSampler(
            env,
            exploration_strategy=es,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            **variant['sampler_params']
        )
        sampler.collect_data()
        if save_replay_buffer:
            sampler.save_replay_buffer()
        return sampler.replay_buffer
    else:
        dataset_path = variant['dataset_path']
        with open(dataset_path, 'rb') as handle:
            return pickle.load(handle)
