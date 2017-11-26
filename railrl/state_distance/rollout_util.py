import numpy as np

from railrl.policies.state_distance import UniversalPolicy
from railrl.samplers.util import rollout


class MultigoalSimplePathSampler(object):
    def __init__(
            self,
            env,
            policy,
            max_samples,
            max_path_length,
            discount_sampling_function,
            goal_sampling_function,
            cycle_taus_for_rollout=True,
    ):
        self.env = env
        self.policy = policy
        self.max_samples = max_samples
        self.max_path_length = max_path_length
        self.discount_sampling_function = discount_sampling_function
        self.goal_sampling_function = goal_sampling_function
        self.cycle_taus_for_rollout = cycle_taus_for_rollout

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        paths = []
        for i in range(self.max_samples // self.max_path_length):
            discount = self.discount_sampling_function()
            goal = self.goal_sampling_function()
            path = multitask_rollout(
                self.env,
                self.policy,
                goal,
                discount,
                max_path_length=self.max_path_length,
                decrement_discount=self.cycle_taus_for_rollout,
                cycle_tau=self.cycle_taus_for_rollout,
            )
            path_length = len(path['observations'])
            path['goal_states'] = expand_goal(goal, path_length)
            paths.append(path)
        return paths


def expand_goal(goal, path_length):
    return np.repeat(
        np.expand_dims(goal, 0),
        path_length,
        0,
    )


def multitask_rollout(
        env,
        agent: UniversalPolicy,
        goal,
        discount,
        max_path_length=np.inf,
        animated=False,
        decrement_discount=False,
        cycle_tau=False,
):
    env.set_goal(goal)
    agent.set_goal(goal)
    agent.set_discount(discount)
    if decrement_discount:
        assert max_path_length >= discount
        path = rollout_decrement_tau(
            env,
            agent,
            discount,
            max_path_length=max_path_length,
            animated=animated,
            cycle_tau=cycle_tau,
        )
    else:
        path = rollout(
            env,
            agent,
            max_path_length=max_path_length,
            animated=animated,
        )
    goal_expanded = np.expand_dims(goal, axis=0)
    # goal_expanded.shape == 1 x goal_dim
    path['goal_states'] = goal_expanded.repeat(len(path['observations']), 0)
    # goal_states.shape == path_length x goal_dim
    return path


def rollout_decrement_tau(env, agent, init_tau, max_path_length=np.inf,
                          animated=False, cycle_tau=False):
    """
    Decrement tau by one at each time step. If tau < 0, keep it at zero or
    reset it to the init tau.

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param cycle_tau: If False, just keep tau equal to zero once it reaches
    zero. Otherwise cycle it.
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    taus = []
    o = env.reset()
    next_o = None
    path_length = 0
    tau = init_tau
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        agent.set_discount(tau)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        taus.append(tau)
        path_length += 1
        tau -= 1
        if tau < 0:
            if cycle_tau:
                tau = init_tau
            else:
                tau = 0
        if d:
            break
        o = next_o
        if animated:
            env.render()
            # input("Press Enter to continue...")

    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
        final_observation=next_o,
        taus=np.array(taus),
    )