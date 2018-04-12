from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv
from railrl.envs.wrappers import ImageMujocoEnv
import cv2
import numpy as np
from mujoco_py.builder import cymj

print("making env")
env = SawyerXYZEnv()
print("starting rollout")
while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    for t in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print("action", action)
        if done:
            break
    print("new episode")
