"""Uses a spacemouse as action input into the environment.

To use this, first clone robosuite (git@github.com:anair13/robosuite.git),
add it to the python path, and ensure you can run the following file (and
see input values from the spacemouse):

robosuite/devices/spacemouse.py

You will likely have to `pip install hidapi` and Spacemouse drivers.
"""

import os
import shutil
import time
import argparse
import datetime
# import h5py
# from glob import glob
import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv

if __name__ == '__main__':
    # env = MultiSawyerEnv(object_meshes=None, num_objects=3,
    #     finger_sensors=False, do_render=False, fix_z=True,
    #     fix_gripper=True, fix_rotation=True)
    env = MultiSawyerEnv(
        do_render=False,
        finger_sensors=False,
        num_objects=5,
        object_meshes=None,
        workspace_low = np.array([-0.25, 0.45, 0.05]),
        workspace_high = np.array([0.25, 0.95, 0.45]),
        hand_low = np.array([-0.25, 0.45, 0.05]),
        hand_high = np.array([0.25, 0.95, 0.45]),
        fix_z=False,
        fix_gripper=True,
        fix_rotation=True,
        cylinder_radius=0.02,
        maxlen=0.06,
    )
    # env = ImageEnv(env,
    #     non_presampled_goal_img_is_garbage=True,
    #     recompute_reward=False,
    #     init_camera=sawyer_pusher_camera_upright_v2,
    # )
    # env.set_goal(env.sample_goals(1))
    for i in range(10000):
        print(i)

        # a[:3] = np.array((0, 0.7, 0.1)) - env.get_endeff_pos()
        # a = np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0.1, 0 ,  1])
        a = np.random.uniform(-1, 1, 3)
        o, _, _, _ = env.step(a)
        if i % 1000 == 0:
            env.reset()
        # print(env.sim.data.qpos[:7])
        env.render()

        # img = o["image_observation"].reshape((84, 84, 3))
        # print(img.shape)
        # cv2.imshow('window', img)
        # cv2.waitKey(10)
