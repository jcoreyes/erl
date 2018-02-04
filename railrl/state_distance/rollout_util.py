import numpy as np

from railrl.state_distance.policies import UniversalPolicy


class MultigoalSimplePathSampler(object):
    def __init__(
            self,
            env,
            policy,
            max_samples,
            max_path_length,
            tau_sampling_function,
            goal_sampling_function,
            cycle_taus_for_rollout=True,
    ):
        self.env = env
        self.policy = policy
        self.max_samples = max_samples
        self.max_path_length = max_path_length
        self.tau_sampling_function = tau_sampling_function
        self.goal_sampling_function = goal_sampling_function
        self.cycle_taus_for_rollout = cycle_taus_for_rollout

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        paths = []
        for i in range(self.max_samples // self.max_path_length):
            tau = self.tau_sampling_function()
            goal = self.goal_sampling_function()
            path = multitask_rollout(
                self.env,
                self.policy,
                goal,
                tau,
                max_path_length=self.max_path_length,
                decrement_tau=self.cycle_taus_for_rollout,
                cycle_tau=self.cycle_taus_for_rollout,
            )
            path_length = len(path['terminals'])
            path['goals'] = _expand_goal(goal, path_length)
            paths.append(path)
        return paths


def _expand_goal(goal, path_length):
    return np.repeat(
        np.expand_dims(goal, 0),
        path_length,
        0,
    )


def multitask_rollout(
        env,
        agent: UniversalPolicy,
        goal,
        init_tau,
        max_path_length=np.inf,
        animated=False,
        decrement_tau=False,
        cycle_tau=False,
        get_action_kwargs=None,
):

    if get_action_kwargs is None:
        get_action_kwargs = {}
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
    if animated:
        env.render()

    tau = init_tau
    env.set_goal(goal)
    agent.set_goal(goal)
    agent.set_tau(tau)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        agent.set_tau(tau)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        taus.append(np.array([tau]))
        path_length += 1
        if decrement_tau:
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
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
        num_steps_left=np.array(taus),
        goals=_expand_goal(goal, len(terminals))
    )