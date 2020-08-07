import argparse

import numpy as np

from rlkit.envs.pygame.water_maze import (
    WaterMaze,
    WaterMaze1D,
    WaterMazeEasy1D,
    WaterMazeMemory1D,
    WaterMazeHard,
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy

parser = argparse.ArgumentParser()
parser.add_argument("--small", action='store_true', help="Use a small maze.")
parser.add_argument("--nreset", default=0, type=int,
                    help="# steps until teleport.")
parser.add_argument("--H", default=20, type=int, help="env horizon.")
parser.add_argument("--render", action='store_true', help="Render env.")
args = parser.parse_args()

# env = WaterMaze()
env = WaterMazeHard()
# env = WaterMaze1D()
# env = WaterMazeMemory1D()

all_returns = []
es = OUStrategy(env.action_space)
print(args.H)
while True:
    obs = env.reset()
    es.reset()
    # print("init obs", obs)
    zero_action = np.zeros(2)
    action = zero_action
    last_reward_t = 0
    print("---------- RESET ----------")
    returns = 0
    for t in range(args.H):
        # action = es.get_action_from_raw_action(zero_action)
        obs, reward, done, info = env.step(action)
        # print("action", action)
        # print("obs", obs)
        target = info['target_position']
        # print("target", target)
        returns += reward
        # time.sleep(0.1)
        if reward > 0:
            time_to_goal = t - last_reward_t
            if time_to_goal > 1:
                # print("Time to goal", time_to_goal)
                last_reward_t = t
        delta = obs[:2] - target
        action = - delta * 10
        # action = np.clip(action, -1, 1)
        if args.render:
            env.render()
    print("Returns", returns)
    all_returns.append(returns)
    print("Returns Mean", np.mean(all_returns))
    print("Returns Std", np.std(all_returns, axis=0))
