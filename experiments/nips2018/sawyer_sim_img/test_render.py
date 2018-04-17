from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv
from railrl.envs.wrappers import ImageMujocoEnv
import cv2
import numpy as np
from mujoco_py.builder import cymj

from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.policies.simple import ZeroPolicy

print("making env")
env = SawyerXYZEnv()
policy = ZeroPolicy(env.action_space.low.size)
es = OUStrategy(
    env.action_space,
    theta=1
)
policy = exploration_policy = PolicyWrappedWithExplorationStrategy(
    exploration_strategy=es,
    policy=policy,
)
print("starting rollout")
while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    for t in range(1000):
        # action = env.action_space.sample()*10
        action, _ = policy.get_action(None)
        if (t//100) % 2 == 0:
            action[3] = -10
        else:
            action[3] = 10
        obs, reward, done, info = env.step(action)
        env.render()
        print("action", action)
        if done:
            break
    print("new episode")
