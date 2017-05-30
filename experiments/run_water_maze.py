from railrl.envs.water_maze import WaterMaze
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--small", action='store_true', help="Use a small maze.")
args = parser.parse_args()

env = WaterMaze(use_small_maze=args.small)

while True:
    env.reset()
    action = np.zeros(2)
    last_reward_t = 0
    print("---------- RESET ----------")
    for t in range(100):
        obs, reward, done, info = env.step(action)
        if reward > 0:
            time_to_goal = t - last_reward_t
            if time_to_goal > 1:
                print("Time to goal", time_to_goal)
                last_reward_t = t
        delta = obs[:2] - info['target_position']
        action = - delta * 10
        env.render()
