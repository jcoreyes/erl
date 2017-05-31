from railrl.envs.water_maze import WaterMaze, WaterMazeMemory
import numpy as np
import argparse

from railrl.exploration_strategies.ou_strategy import OUStrategy

parser = argparse.ArgumentParser()
parser.add_argument("--small", action='store_true', help="Use a small maze.")
args = parser.parse_args()

env = WaterMazeMemory(use_small_maze=args.small, include_velocity=True)

all_returns = []
es = OUStrategy(env)
while True:
    obs = env.reset()
    es.reset()
    # print("init obs", obs)
    zero_action = np.zeros(2)
    last_reward_t = 0
    print("---------- RESET ----------")
    returns = 0
    for t in range(200):
        action = es.get_action_from_raw_action(zero_action)
        obs, reward, done, info = env.step(action)
        returns += reward
        # print("obs", obs)
        # time.sleep(0.1)
        if reward > 0:
            time_to_goal = t - last_reward_t
            if time_to_goal > 1:
                # print("Time to goal", time_to_goal)
                last_reward_t = t
        delta = obs[:2] - info['target_position']
        # action = - delta * 10
        # env.render()
    print("Returns", returns)
    all_returns.append(returns)
    print("Returns Mean", np.mean(all_returns))
    print("Returns Std", np.std(all_returns, axis=0))
