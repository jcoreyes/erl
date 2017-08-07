from railrl.envs.ros.sawyer_env import SawyerEnv
import numpy as np

def create_action(mag):
    action = np.random.rand((1, 7))[0]

env = SawyerEnv('right', None, safe_reset=True)

env.reset()

for _ in range(10):
    action = create_action(.1)
    env.step(action)

