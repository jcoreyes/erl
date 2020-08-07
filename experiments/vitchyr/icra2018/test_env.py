import argparse

import numpy as np

from rlkit.envs.mujoco.pusher_avoider_3dof import PusherAvoiderEnv3DOF
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
parser.add_argument("--H", default=100, type=int, help="env horizon.")
parser.add_argument("--render", action='store_true', help="Render env.")
args = parser.parse_args()

# env = WaterMaze()
env = WaterMazeHard()
# env = WaterMaze1D()
# env = WaterMazeMemory1D()

all_returns = []
es = OUStrategy(env.action_space)
print(args.H)
i = 0
env = PusherAvoiderEnv3DOF(
    task='both',
    init_config=i%5,
)
for _ in range(5):
    i += 1
    obs = env.reset()
    es.reset()
    # print("init obs", obs)
    zero_action = np.zeros(3)
    action = zero_action
    last_reward_t = 0
    print("---------- RESET ----------")
    returns = 0
    for t in range(args.H):
        obs, reward, done, info = env.step(action)
        if args.render:
            env.render()
    print("Returns", returns)
    all_returns.append(returns)
    print("Returns Mean", np.mean(all_returns))
    print("Returns Std", np.std(all_returns, axis=0))
while True:
    pass
