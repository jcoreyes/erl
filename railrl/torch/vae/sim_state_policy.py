from railrl.envs.remote import RemoteRolloutEnv
from railrl.envs.vae_wrappers import VAEWrappedEnv
from railrl.misc.eval_util import get_generic_path_information
from railrl.samplers.util import rollout
from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from railrl.core import logger

import os.path as osp
import os

filename = str(uuid.uuid4())

import numpy as np
import time

def rollout(env, agent, max_path_length=np.inf, animated=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    goal = env.get_goal()
    agent.reset()
    # obs, goal = env.goal_obs, env.cur_obs
    next_o = None
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(obs)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def sim(env, policy, filename, horizon=100, animated=True):
    policy.train(False) # is this right/necessary?
    while True:
        paths = []
        for _ in range(1):
            goal = env.sample_goal_for_rollout()
            env.set_goal(goal)
            path = rollout(
                env,
                policy,
                max_path_length=horizon,
                animated=True,
            )
            print("last state", path['next_observations'][-1][21:24])
            paths.append(path)
        env.log_diagnostics(paths)
        for key, value in get_generic_path_information(paths).items():
            logger.record_tabular(key, value)
        logger.dump_tabular()


def simulate_policy(args):
    data = joblib.load(args.file)
    if 'eval_policy' in data:
        policy = data['eval_policy']
    elif 'policy' in data:
        policy = data['policy']
    elif 'exploration_policy' in data:
        policy = data['exploration_policy']
    elif 'naf_policy' in data:
        policy = data['naf_policy']
    elif 'optimizable_qfunction' in data:
        qf = data['optimizable_qfunction']
        policy = qf.implicit_policy
    else:
        raise Exception("No policy found in loaded dict. Keys: {}".format(
            data.keys()
        ))

    env = data['env']
    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")

    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
        if hasattr(env, "vae"):
            env.vae.cuda()
    else:
        # make sure everything is on the CPU
        set_gpu_mode(False)
        policy.cpu()
        if hasattr(env, "vae"):
            env.vae.cpu()

    if args.pause:
        import ipdb; ipdb.set_trace()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    paths = sim(env, policy, filename, horizon=args.H)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
