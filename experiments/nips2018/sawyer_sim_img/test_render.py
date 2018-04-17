from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv
import numpy as np

print("making env")
env = SawyerXYZEnv()
print("starting rollout")
while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    for t in range(1000):
        action = env.action_space.sample()
        action = np.hstack([action, np.array([0])])
        action[0] -= 0.5
        obs, reward, done, info = env.step(action)
        env.render()
        print("action", action)
        if done:
            break
    print("new episode")
