from math import pi
import numpy as np
from railrl.envs.multitask.reacher_env import MultitaskReacherEnv

env = MultitaskReacherEnv()
env.reset()
qpos = np.array([
    pi/2,  # arm-x
    pi/4,
    0.1,  # target-x
    -0.1,
])
qvel = np.zeros(4)
env.set_state(qpos, qvel)
print(env.get_body_com('fingertip'))
for _ in range(1000):
    env.render()
