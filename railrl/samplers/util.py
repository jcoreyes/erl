import numpy as np

def rollout(env, agent, max_path_length=np.inf, animated=False):
    """
    Like rllab's rollout, but do not flatten actions/observations.
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
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
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

    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
    )
