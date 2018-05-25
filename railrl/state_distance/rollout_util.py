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
            render=False,
            env_samples_goal_on_reset=False,
    ):
        self.env = env
        self.policy = policy
        self.max_samples = max_samples
        self.max_path_length = max_path_length
        self.tau_sampling_function = tau_sampling_function
        self.goal_sampling_function = goal_sampling_function
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self.render = render
        self.env_samples_goal_on_reset = env_samples_goal_on_reset

    def obtain_samples(self):
        paths = []
        for i in range(self.max_samples // self.max_path_length):
            tau = self.tau_sampling_function()
            if self.env_samples_goal_on_reset:
                goal = None
            else:
                goal = self.goal_sampling_function()
            path = multitask_rollout(
                self.env,
                self.policy,
                tau,
                goal=goal,
                max_path_length=self.max_path_length,
                decrement_tau=self.cycle_taus_for_rollout,
                cycle_tau=self.cycle_taus_for_rollout,
                animated=self.render,
                env_samples_goal_on_reset=self.env_samples_goal_on_reset,
            )
            paths.append(path)
        return paths

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("could not import matplotlib")
ax1 = None
ax2 = None


def debug(env, obs, agent_info):
    global ax1
    global ax2
    if ax1 is None:
        _, (ax1, ax2) = plt.subplots(1, 2)

    subgoal_seq = agent_info['subgoal_seq']
    planned_action_seq = agent_info['planned_action_seq']
    real_obs_seq = env.true_states(
        obs, planned_action_seq
    )
    ax1.clear()
    env.plot_trajectory(
        ax1,
        np.array(subgoal_seq),
        np.array(planned_action_seq),
        goal=env._target_position,
    )
    ax1.set_title("imagined")
    ax2.clear()
    env.plot_trajectory(
        ax2,
        np.array(real_obs_seq),
        np.array(planned_action_seq),
        goal=env._target_position,
    )
    ax2.set_title("real")
    plt.draw()
    plt.pause(0.001)


def multitask_rollout(
        env,
        agent: UniversalPolicy,
        init_tau,
        max_path_length=np.inf,
        goal=None,
        animated=False,
        decrement_tau=False,
        cycle_tau=False,
        get_action_kwargs=None,
        env_samples_goal_on_reset=False,
):
    if get_action_kwargs is None:
        get_action_kwargs = {}
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    taus = []

    agent.reset()
    path_length = 0
    if animated:
        env.render()

    tau = np.array([init_tau])
    if goal is None and env_samples_goal_on_reset:
        o = env.reset()
        goal = env.get_goal()
    else:
        if goal is None:
            goal = env.sample_goal_for_rollout()
        env.set_goal(goal)
        o = env.reset()
    assert (env.get_goal() == goal).all()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o, goal, tau, **get_action_kwargs)
        if animated:
            env.render()
        next_o, r, d, env_info = env.step(a)
        next_observations.append(next_o)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        taus.append(tau.copy())
        path_length += 1
        if decrement_tau:
            tau -= 1
        if tau < 0:
            if cycle_tau:
                tau = np.array([init_tau])
            else:
                tau = np.array([0])
        if d:
            break
        o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)

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


def _expand_goal(goal, path_length):
    return np.repeat(
        np.expand_dims(goal, 0),
        path_length,
        0,
    )
