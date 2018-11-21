"""Uses a spacemouse as action input into the environment.

To use this, first clone robosuite (git@github.com:anair13/robosuite.git),
add it to the python path, and ensure you can run the following file (and
see input values from the spacemouse):

robosuite/devices/spacemouse.py

You will likely have to `pip install hidapi` and Spacemouse drivers.
"""

from robosuite.devices import SpaceMouse

import os
import shutil
import time
import argparse
import datetime
# import h5py
# from glob import glob
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T

from multiworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

import cv2

def collect_one_rollout(env):
    o = env.reset()
    traj = dict(obs=[o], actions=[])

    while True:
        state = device.get_controller_state()
        dpos, rotation, accept, reset = (
            state["dpos"],
            state["rotation"],
            state["left_click"],
            state["right_click"],
        )
        a = dpos

        o, r, _, info = env.step(a)

        traj["obs"].append(o)
        traj["actions"].append(a)
        traj["rewards"].append(r)

        # env.render()
        img = o["image_observation"].reshape((84, 84, 3))
        cv2.imshow('window', img)
        cv2.waitKey(10)

        if reset or accept:
            return accept, traj

def collect_one_rollout_goal_conditioned(env):
    goal = env.sample_goals(1)
    env.set_to_goal(goal)
    goal_obs = env._get_obs()
    goal_image = goal_obs["image_observation"].reshape((84, 84, 3))
    cv2.imshow('goal', goal_image)
    cv2.waitKey(10)

    o = env.reset()
    env.set_goal(goal)
    traj = dict(
        observations=[o],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
        goal=goal, goal_obs=goal_obs)

    while True:
        state = device.get_controller_state()
        dpos, rotation, accept, reset = (
            state["dpos"],
            state["rotation"],
            state["left_click"],
            state["right_click"],
        )
        a = dpos
        a[1] *= -1

        traj["observations"].append(o)

        o, r, done, info = env.step(a)

        traj["actions"].append(a)
        traj["rewards"].append(r)
        traj["next_observations"].append(o)
        traj["terminals"].append(done)
        traj["agent_infos"].append(info)
        traj["env_infos"].append(info)

        # env.render()
        img = o["image_observation"].reshape((84, 84, 3))
        cv2.imshow('window', img)
        cv2.waitKey(100)

        if reset or accept:
            return accept, traj

def collect_demos(env, path="demos.npy", N=10):
    data = []

    while len(data) < N:
        accept, traj = collect_one_rollout_goal_conditioned(env)
        if accept:
            data.append(traj)
            print("accepted trajectory length", len(traj["observations"]))
            print("last reward", traj["rewards"][-1])
            print("accepted", len(data), "trajectories")
        else:
            print("discarded trajectory")

    np.save(path, data)

if __name__ == '__main__':
    device = SpaceMouse()

    # env = MultiSawyerEnv(object_meshes=None, num_objects=3,
    #     finger_sensors=False, do_render=False, fix_z=True,
    #     fix_gripper=True, fix_rotation=True)
    size = 0.1
    low = np.array([-size, 0.4 - size, 0])
    high = np.array([size, 0.4 + size, 0.1])
    env = MultiSawyerEnv(
        do_render=False,
        finger_sensors=False,
        num_objects=1,
        object_meshes=None,
        workspace_low = low,
        workspace_high = high,
        hand_low = low,
        hand_high = high,
        fix_z=True,
        fix_gripper=True,
        fix_rotation=True,
        cylinder_radius=0.03,
        maxlen=0.03,
        init_hand_xyz=(0, 0.4-size, 0.089),
    )
    env = ImageEnv(env,
        non_presampled_goal_img_is_garbage=True,
        recompute_reward=False,
        init_camera=sawyer_pusher_camera_upright_v2,
    )
    # env.set_goal(env.sample_goals(1))

    collect_demos(env)

