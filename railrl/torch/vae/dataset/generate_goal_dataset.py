import cv2
import railrl.torch.pytorch_util as ptu
from multiworld.core.image_env import ImageEnv
from railrl.envs.vae_wrappers import VAEWrappedEnv
from railrl.misc.asset_loader import load_local_or_remote_file
from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
import numpy as np
from railrl.policies.simple import RandomPolicy
import os.path as osp


def generate_goal_data_set(env=None, num_goals=1000, use_cached_dataset=False,
                           action_scale=1 / 10):
    if use_cached_dataset and osp.isfile(
            '/tmp/goals' + str(num_goals) + '.npy'):
        goal_dict = np.load('/tmp/goals' + str(num_goals) + '.npy').item()
        print("loaded data from saved file")
        return goal_dict
    cached_goal_keys = [
        'latent_desired_goal',
        'image_desired_goal',
        'state_desired_goal',
        'joint_desired_goal',
    ]
    goal_sizes = [
        env.observation_space.spaces['latent_desired_goal'].low.size,
        env.observation_space.spaces['image_desired_goal'].low.size,
        env.observation_space.spaces['state_desired_goal'].low.size,
        7
    ]
    observation_keys = [
        'latent_observation',
        'image_observation',
        'state_observation',
        'state_observation',
    ]
    goal_generation_dict = dict()
    for goal_key, goal_size, obs_key in zip(
            cached_goal_keys,
            goal_sizes,
            observation_keys,
    ):
        goal_generation_dict[goal_key] = [goal_size, obs_key]
    goal_dict = dict()
    policy = RandomPolicy(env.action_space)
    es = OUStrategy(action_space=env.action_space, theta=0)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    for goal_key in goal_generation_dict:
        goal_size, obs_key = goal_generation_dict[goal_key]
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
            goal_size, obs_key = goal_generation_dict[goal_key]
            goal_dict[goal_key][i, :] = obs[obs_key]
    np.save('/tmp/goals' + str(num_goals) + '.npy', goal_dict)
    return goal_dict

def generate_goal_dataset_using_policy(
        env=None,
        num_goals=1000,
        use_cached_dataset=False,
        policy_file=None,
        show=False,
        path_length=500,
        save_file_prefix=None,
        env_id=None,
        tag='',
):
    env_class = type(env.wrapped_env.wrapped_env)
    if save_file_prefix is None and env_id is not None:
        save_file_prefix = env_id
    elif save_file_prefix is None:
        save_file_prefix = env_class.__name__
    filename = "/tmp/{}_N{}_imsize{}goals{}.npy".format(
        save_file_prefix,
        str(num_goals),
        env.imsize,
        tag,
    )
    if use_cached_dataset and osp.isfile(filename):
        goal_dict = np.load(filename).item()
        print("Loaded data from {}".format(filename))
        return goal_dict

    goal_generation_dict = dict()
    for goal_key, obs_key in [
        ('image_desired_goal', 'image_achieved_goal'),
        ('state_desired_goal', 'state_achieved_goal'),
    ]:
        goal_size = env.observation_space.spaces[goal_key].low.size
        goal_generation_dict[goal_key] = [goal_size, obs_key]

    goal_dict = dict()
    policy_file = load_local_or_remote_file(policy_file)
    policy = policy_file['policy']
    if ptu.gpu_enabled():
        policy.cuda()
    for goal_key in goal_generation_dict:
        goal_size, obs_key = goal_generation_dict[goal_key]
        goal_dict[goal_key] = np.zeros((num_goals, goal_size))
    print('Generating Random Goals')
    for j in range(num_goals):
        obs = env.reset()
        policy.reset()
        for i in range(path_length):
            policy_obs = np.hstack((
                obs['state_observation'],
                obs['state_desired_goal'],
            ))
            action, _ = policy.get_action(policy_obs)
            obs, _, _, _ = env.step(action)
        if show:
            img = obs['image_observation']
            img = img.reshape(3, env.imsize, env.imsize).transpose()
            img = img[::-1, :, ::-1]
            cv2.imshow('img', img)
            cv2.waitKey(1)

        for goal_key in goal_generation_dict:
            goal_size, obs_key = goal_generation_dict[goal_key]
            goal_dict[goal_key][j, :] = obs[obs_key]
    np.save(filename, goal_dict)
    print("Saving file to {}".format(filename))
    return goal_dict

def generate_goal_dataset_using_set_to_goal(
        env=None,
        num_goals=1000,
        use_cached_dataset=False,
        show=False,
        save_filename=None,
):
    filename = save_filename or '/tmp/goals_n{}_{}.npy'.format(num_goals, env)
    if use_cached_dataset and osp.isfile(filename):
        goal_dict = np.load(filename).item()
        print("Loaded data from {}".format(filename))
        return goal_dict

    goal_generation_dict = dict()
    for goal_key, obs_key in [
        ('image_desired_goal', 'image_achieved_goal'),
        ('state_desired_goal', 'state_achieved_goal'),
    ]:
        goal_size = env.observation_space.spaces[goal_key].low.size
        goal_generation_dict[goal_key] = [goal_size, obs_key]

    goal_dict = dict()
    for goal_key in goal_generation_dict:
        goal_size, obs_key = goal_generation_dict[goal_key]
        goal_dict[goal_key] = np.zeros((num_goals, goal_size))
    print('Generating Random Goals')
    for j in range(num_goals):
        if type(env) == VAEWrappedEnv:
            goal = env.wrapped_env.wrapped_env.sample_goal()
        elif type(env) == ImageEnv:
            goal = env.wrapped_env.sample_goal()
        else:
            goal = env.sample_goal()
        env.set_to_goal(goal)
        obs = env._get_obs()
        if show:
            img = obs['image_observation']
            img = img.reshape(3, env.imsize, env.imsize).transpose()
            img = img[::-1, :, ::-1]
            cv2.imshow('img', img)
            cv2.waitKey(1)
        for goal_key in goal_generation_dict:
            goal_size, obs_key = goal_generation_dict[goal_key]
            goal_dict[goal_key][j, :] = obs[obs_key]
    np.save(filename, goal_dict)
    print("Saving file to {}".format(filename))
    return goal_dict
