import argparse
import numpy as np

from rlkit.envs.multitask.reacher_env import XyMultitaskReacherEnv

env = XyMultitaskReacherEnv()


def set_state(target_pos, joint_angles, velocity):
    qpos = np.concatenate([joint_angles, target_pos])
    qvel = np.array([velocity[0], velocity[1], 0, 0])
    env.reset()
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

    velocity = obs[6:8]
    set_state(target_pos, joint_angles, velocity)
    env.do_simulation(action, env.frame_skip)
    pos = env.get_body_com('fingertip')[:2]
    return -np.linalg.norm(pos - target_pos)


def sample_best_action_ground_truth(obs, num_samples):
    sampled_actions = np.random.uniform(-.1, .1, size=(num_samples, 2))
    # resolution = 10
    # x = np.linspace(-1, 1, resolution)
    # y = np.linspace(-1, 1, resolution)
    # sampled_actions = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    q_values = [true_q(obs[-2:], obs, a) for a in sampled_actions]
    max_i = np.argmax(q_values)
    return sampled_actions[max_i]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=float, default=0.1)
    parser.add_argument('y', type=float, default=0.1)
    args = parser.parse_args()
    goal = np.array([args.x, args.y])
    num_samples = 10

    eval_env = XyMultitaskReacherEnv()
    obs = eval_env.reset()
    for _ in range(1000):
        new_obs = np.hstack((obs, goal))
        action = sample_best_action_ground_truth(new_obs, num_samples)
        obs, r, d, env_info = eval_env.step(action)
        eval_env.render()
