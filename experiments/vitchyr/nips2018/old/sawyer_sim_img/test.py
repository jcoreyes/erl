from rlkit.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv
from rlkit.envs.wrappers import ImageMujocoEnv
import cv2
import numpy as np

print("making env")
sawyer = SawyerXYZEnv()
env = ImageMujocoEnv(sawyer, imsize=400)

print("starting rollout")
while True:
    obs = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        raw_img = env._image_observation()
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
