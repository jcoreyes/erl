from rlkit.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv, SawyerEnv, SawyerBlockEnv
from rlkit.envs.wrappers import ImageEnv
import numpy as np
import cv2

print("making env")
env = SawyerBlockEnv()
# env = SawyerEnv()
env = ImageEnv(env, imsize=4)

print("starting rollout")
while True:
    # import pdb; pdb.set_trace()
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    x = np.random.rand((4))
    for t in range(100):
        print(t)
        # action = env.action_space.sample()
        action = np.random.rand((4))
        # action = np.zeros((4))
        # action = x
        obs, reward, done, info = env.step(action)
        env.render()
        # print(obs)
        # cv2.imshow('obs', obs)
        # cv2.waitKey(1)
        if done:
            break
    print("new episode")
