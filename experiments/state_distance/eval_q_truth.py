import numpy as np

from railrl.envs.multitask.reacher_env import MultitaskReacherEnv

env = MultitaskReacherEnv()


def set_state(target_pos, joint_angles):
    qpos, qvel = np.concatenate([joint_angles, target_pos]), np.zeros(4)
    env.set_state(qpos, qvel)


def true_q(target_pos, obs, action):
    c1 = obs[0]  # cosine of angle 1
    c2 = obs[1]
    s1 = obs[2]
    s2 = obs[3]

    joint_angles = np.array([
        np.arctan2(s1, c1),
        np.arctan2(s2, c2),
    ])

    set_state(target_pos, joint_angles)
    env.do_simulation(action, env.frame_skip)
    pos = env.get_body_com('fingertip')[:2]
    return -np.linalg.norm(pos - target_pos)


def sample_best_action_ground_truth(obs, num_samples):
    sampled_actions = np.random.uniform(-.1, .1, size=(num_samples, 2))
    q_values = [true_q(obs[-2:], obs, a) for a in sampled_actions]
    max_i = np.argmax(q_values)
    return sampled_actions[max_i]

if __name__ == "__main__":
    goal = np.array([.1, .1])
    num_samples = 10

    obs = env.reset()
    for _ in range(1000):
        new_obs = np.hstack((obs, goal))
        action = sample_best_action_ground_truth(new_obs, num_samples)
        obs, r, d, env_info = env.step(action)
        env.render()
