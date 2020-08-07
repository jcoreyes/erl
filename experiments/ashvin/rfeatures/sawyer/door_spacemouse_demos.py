from rlkit.demos.collect_demo import collect_demos_fixed
from rlkit.demos.spacemouse.input_server import SpaceMouseExpert

from multiworld.core.image_env import ImageEnv
# from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerMultiobjectEnv
# from multiworld.envs.pygame.point2d import Point2DWallEnv

import gym
import numpy as np

# from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from multiworld.envs.real_world.sawyer.sawyer_reaching import SawyerReachXYZEnv

import time
import rospy

# from sawyer_control.envs.sawyer_insertion_refined_USB_sparse_RLonly import SawyerHumanControlEnv

if __name__ == '__main__':
    scale = 1.0
    expert = SpaceMouseExpert(
        xyz_dims=3,
        xyz_remap=[0, 1, 2],
        xyz_scale=[-scale, -scale, scale],
    )

    # env = gym.make("MountainCarContinuous-v0")
    # env = SawyerHumanControlEnv(action_mode='joint_space_impd', position_action_scale=1, max_speed=0.015)
    env = SawyerReachXYZEnv(action_mode="position", max_speed = 0.05, camera="sawyer_head")

    env = ImageEnv(env,
        recompute_reward=False,
        transpose=True,
        image_length=450000,
        reward_type="image_distance",
        # init_camera=sawyer_pusher_camera_upright_v2,
    )

    # env.reset()

    for i in range(25):
        collect_demos_fixed(env, expert, "/home/anair/ros_ws/src/railrl-private/demos/demo_v4_grey_%i.pkl" %i, 1, horizon=1000, pause=0.05)
        print("Collected demo: ", i)
    # for i in range(10):
    # collect_demos_fixed(env, expert, "demos/demo_v3.pkl", 1, horizon=1000, pause=0.05)

    # o = None
    # while True:
    #     a, valid, reset, accept = expert.get_action(o)

    #     if valid:
    #         o, r, done, info = env.step(a)
    #         time.sleep(0.05)

    #     if reset or accept:
    #         env.reset()

    #     if rospy.is_shutdown():
    #         break

    #     time.sleep(0.01)

    # exit()