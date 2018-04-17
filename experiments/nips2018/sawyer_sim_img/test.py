from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv
from railrl.envs.wrappers import ImageEnv
import cv2
import numpy as np
from mujoco_py.builder import cymj
import mujoco_py

print("making env")
sawyer = SawyerXYZEnv()
sim = sawyer.sim
viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
viewer.cam.trackbodyid = 0
viewer.cam.distance = 1.0

# robot view
# rotation_angle = 90
# cam_dist = 1
# cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

# 3rd person view
cam_dist = 0.2
rotation_angle = 270
cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

for i in range(3):
    viewer.cam.lookat[i] = cam_pos[i]
viewer.cam.distance = cam_pos[3]
viewer.cam.elevation = cam_pos[4]
viewer.cam.azimuth = cam_pos[5]
viewer.cam.trackbodyid = -1
sim.add_render_context(viewer)
env = ImageEnv(sawyer, imsize=400)

class Viewer(cymj.MjRenderContextWindow):
    def __init__(self, sim):
        super().__init__(sim)

import ipdb; ipdb.set_trace()
print("starting rollout")
while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    for t in range(100):
        action = env.action_space.sample()
        action = np.hstack([action, np.array([0])])
        obs, reward, done, info = env.step(action)
        # env.render()
        img = np.concatenate((
            obs[::-1, :, 2:3],
            obs[::-1, :, 1:2],
            obs[::-1, :, 0:1],
        ), axis=2)
        cv2.imshow('obs', img)
        cv2.waitKey(1)
        # if done:
        #     break
    print("new episode")
