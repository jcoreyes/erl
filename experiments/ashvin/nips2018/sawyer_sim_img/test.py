from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv, SawyerEnv
from railrl.envs.wrappers import ImageEnv
import cv2

print("making env")
# env = SawyerXYZEnv()
env = SawyerEnv()
env = ImageEnv(env, imsize=400)

print("starting rollout")
while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    for t in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # env.render()
        print(obs)
        cv2.imshow('obs', obs)
        cv2.waitKey(1)
        if done:
            break
    print("new episode")
