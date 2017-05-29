from railrl.envs.water_maze import WaterMaze
import numpy as np

env = WaterMaze()

while True:
    env.reset()
    for _ in range(100):
        env.step(np.random.rand(2))
        env.render()
