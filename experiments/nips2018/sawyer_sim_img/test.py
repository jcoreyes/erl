from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv
from railrl.envs.wrappers import ImageMujocoEnv
import cv2
import numpy as np
from mujoco_py.builder import cymj

print("making env")
sawyer = SawyerXYZEnv()
env = ImageMujocoEnv(sawyer, imsize=400)

class Viewer(cymj.MjRenderContextWindow):
    def __init__(self, sim):
        super().__init__(sim)

print("starting rollout")
while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    for t in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        raw_img = env.image_observation()
        img = np.concatenate((
            raw_img[::-1, :, 2:3],
            raw_img[::-1, :, 1:2],
            raw_img[::-1, :, 0:1],
        ), axis=2)
        cv2.imshow('obs', img)
        cv2.waitKey(1)
        # if done:
        #     break
    print("new episode")
