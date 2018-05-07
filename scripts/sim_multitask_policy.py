from railrl.envs.remote import RemoteRolloutEnv
from railrl.samplers.util import rollout
from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from railrl.core import logger
import numpy as np


def multitask_rollout(env, agent, max_path_length=np.inf, animated=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    agent.reset()
    next_o = None
    path_length = 0
    goal = env.sample_goal_for_rollout()
    env.set_goal(goal)
    o = env.reset()
    if animated:
        env.render()
    while path_length < max_path_length:
        goal = env.multitask_goal
        new_o = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_o)
        a = a + np.random.normal(a.shape) / 10
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
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


def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['policy']

    env = data['env']
    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if args.pause:
        import ipdb; ipdb.set_trace()
    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.multitaskpause:
        env.pause_on_goal = True
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    paths = []
    while True:
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
