from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
import numpy as np
from rlkit.policies.simple import RandomPolicy
import os.path as osp

def generate_goal_data_set(env=None, num_goals=1000, use_cached_dataset=False, action_scale=1/10, representation_size=16, imsize=84):
    if use_cached_dataset and osp.isfile('/tmp/goals' + str(num_goals) + '.npy'):
        goal_dict = np.load('/tmp/goals' + str(num_goals) + '.npy').item()
        print("loaded data from saved file")
        return goal_dict
    cached_goal_keys = ['latent_desired_goal', 'image_desired_goal', 'state_desired_goal', 'joint_desired_goal']
    goal_sizes = [representation_size, (imsize ** 2) * 3, 3, 7]
    obs_to_goal_fctns = [lambda x: x, lambda x: x, lambda x: x[-3:], lambda x: x[:7]]
    observation_keys = ['latent_observation', 'image_observation', 'state_observation', 'state_observation']
    goal_generation_dict = dict()
    for goal_key, goal_size, obs_to_goal_fctn, obs_key in zip(cached_goal_keys, goal_sizes, obs_to_goal_fctns,
                                                              observation_keys):
        goal_generation_dict[goal_key] = [goal_size, obs_to_goal_fctn, obs_key]
    goal_dict = dict()
    policy = RandomPolicy(env.action_space)
    es = OUStrategy(action_space=env.action_space, theta=0)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    for goal_key in goal_generation_dict:
        goal_size, obs_to_goal_fn, obs_key = goal_generation_dict[goal_key]
        goal_dict[goal_key] = np.zeros((num_goals, goal_size))
    print('Generating Random Goals')
    for i in range(num_goals):
        if i % 50 == 0:
            print('Reset')
            env.reset_model()
            exploration_policy.reset()
        action = exploration_policy.get_action()[0] * action_scale
        obs, _, _, _ = env.step(
            action
        )
        print(i)
        for goal_key in goal_generation_dict:
            goal_size, obs_to_goal_fn, obs_key = goal_generation_dict[goal_key]
            goal_dict[goal_key][i, :] = obs_to_goal_fn(obs[obs_key])
    np.save('/tmp/goals' + str(num_goals) +'.npy', goal_dict)
    return goal_dict