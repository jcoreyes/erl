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
import time

# from robosuite.devices import SpaceMouse
# import robosuite
# import robosuite.utils.transform_utils as T

# from multiworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv
# from multiworld.core.image_env import ImageEnv
# from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2


import sys
import pickle

### workaround to solve cv2 version conflicts (ROS adds Python2 version of cv2)
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import cv2
# sys.path.insert(0,'/opt/ros/kinetic/lib/python2.7/dist-packages')

# import cv2

class Expert:
    def __init__(self, action_dim=3, **kwargs):
        self.action_dim = action_dim

    def get_action(self, obs):
        a = np.zeros((self.action_dim))
        valid = True
        reset = False
        accept = False
        return (a, valid, reset, accept)

class RandomAgent(Expert):
    def __init__(self, action_scale=0.1, action_space_dim=2):
        self.action_scale = action_scale
        self.action_space_dim = action_space_dim
        self.counter = 0

    def get_action(self, obs):
        action = np.random.uniform(-1, 1, self.action_space_dim)*self.action_scale
        self.counter += 1
        reset = self.counter % 25 == 0
        accept = reset
        valid = True
        # accept = True
        return (action, valid, reset, accept)

class SpaceMouseExpert(Expert):
    def __init__(self, xyz_dims=3, xyz_remap=[0, 1, 2], xyz_scale=[1, 1, 1]):
        """TODO: fill in other params"""

        from robosuite.devices import SpaceMouse

        self.xyz_dims = xyz_dims
        self.xyz_remap = np.array(xyz_remap)
        self.xyz_scale = np.array(xyz_scale)
        self.device = SpaceMouse()

    def get_action(self, obs):
        """Must return (action, valid, reset, accept)"""
        state = self.device.get_controller_state()
        dpos, rotation, accept, reset = (
            state["dpos"],
            state["rotation"],
            state["left_click"],
            state["right_click"],
        )

        xyz = dpos[self.xyz_remap] * self.xyz_scale

        a = xyz[:self.xyz_dims]

        valid = not np.all(np.isclose(a, 0))

        return (a, valid, reset, accept)

def collect_one_rollout(env, expert, horizon=200, render=False, pause=0):
    o = env.reset()

    traj = dict(
        observations=[o],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )

    for i in range(10000):
        a, valid, reset, accept = expert.get_action(o)

        if valid:
            traj["observations"].append(o)

            o, r, done, info = env.step(a)

            traj["actions"].append(a)
            traj["rewards"].append(r)
            traj["next_observations"].append(o)
            traj["terminals"].append(done)
            traj["agent_infos"].append(info)
            traj["env_infos"].append(info)
            print(r)

            if render:
                env.render()

            if pause:
                time.sleep(pause)

        if reset or accept:
            if len(traj["rewards"]) == 0:
                accept = False
            return accept, traj

        time.sleep(0.01)

    return False, []

def draw_grid(img, line_color=(0, 0, 0), thickness=1, type_=None, pxstep=20):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    if type_ is None:
        type_ = cv2.LINE_AA
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

def collect_one_rollout_goal_conditioned(env, expert, horizon=200, threshold=-1):
    # goal = env.sample_goal()
    # env.set_to_goal(goal)
    # goal_obs = env._get_obs()
    # goal_image = goal_obs["image_observation"].reshape((84, 84, 3))
    o = env.reset()

    #changed by MURTAZA
    # goal_image = o["image_desired_goal"].reshape((84, 84, 3)).copy() # .transpose()
    # draw_grid(goal_image)
    # cv2.imshow('goal', goal_image)
    # cv2.waitKey(10)

    # img = o["image_observation"].reshape((84, 84, 3)).copy()
    # # o["image_observation"].reshape((84, 84, 3))
    # draw_grid(img)
    # env.set_goal(goal)
    traj = dict(
        observations=[o],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )

    for i in range(horizon):
        a, _ = expert.get_action(np.concatenate((o['observation'], o['desired_goal'])))

        traj["observations"].append(o)

        o, r, done, info = env.step(a)

        traj["actions"].append(a)
        traj["rewards"].append(r)
        traj["next_observations"].append(o)
        traj["terminals"].append(done)
        traj["agent_infos"].append(info)
        traj["env_infos"].append(info)


            # env.render()
            # img = o["image_observation"].reshape((84, 84, 3)).copy()
            # draw_grid(img)

        # cv2.imshow('window', img)
        # cv2.waitKey(100)

        # if reset or accept:
            # if len(traj["rewards"]) == 0:
                # accept = False
            # return accept, traj
    if threshold == -1:
        accept = True
    elif np.abs(r) < threshold:
        accept = True
    else:
        accept = False
    return accept, traj

def collect_demos(env, expert, path="demos.npy", N=10, horizon=200, threshold=-1):
    data = []

    while len(data) < N:
        accept, traj = collect_one_rollout_goal_conditioned(env, expert, horizon, threshold=threshold)
        if accept:
            data.append(traj)
            print("accepted trajectory length", len(traj["observations"]))
            print("last reward", traj["rewards"][-1])
            print("accepted", len(data), "trajectories")
        else:
            print("discarded trajectory")

    np.save(path, data)


def collect_demos_fixed(env, expert, path="demos.npy", N=10, **kwargs):
    data = []

    i = 0
    while len(data) < N and i < 1000:
        i += 1
        accept, traj = collect_one_rollout(env, expert, **kwargs)
        if accept:
            data.append(traj)
            print("accepted trajectory length", len(traj["observations"]))
            print("last reward", traj["rewards"][-1])
            print("accepted", len(data), "trajectories")
        else:
            print("discarded trajectory")

    # np.save(path, data)
    pickle.dump(data, open(path, "wb"), protocol=2)

if __name__ == '__main__':
    # device = SpaceMouse()
    expert = SpaceMouseExpert()

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

    collect_demos(env, expert)
