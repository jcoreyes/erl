import sys, tty, termios
from railrl.envs.mujoco.sawyer_gripper_env import SawyerPushXYEnv
from railrl.envs.mujoco.sawyer_kitchen import KitchenCabinetEnv

from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.policies.simple import ZeroPolicy
import numpy as np

print("making env")
env = SawyerPushXYEnv(randomize_goals=True, frame_skip=50)
# env = KitchenCabinetEnv()

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

import pygame
from pygame.locals import QUIT, KEYDOWN
pygame.init()

screen = pygame.display.set_mode((400, 300))

char_to_action = {
    'q': np.array([0 , 0 , 1 , 0]),
    'w': np.array([1 , 0 , 0 , 0]),
    'e': np.array([0 , 0 , -1, 0]),
    'a': np.array([0 , 1 , 0 , 0]),
    's': np.array([-1, 0 , 0 , 0]),
    'd': np.array([0 , -1, 0 , 0]),
    'z': np.array([0 , 0 , 0 , 1]),
    'c': np.array([0 , 0 , 0 , -1]),
    'x': np.array([0 , 0 , 0 , 0]),
}

# print("waiting")
# event = pygame.event.wait()
# print("done")


while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    action, _ = policy.get_action(None)
    while True:
        # char = getch()
        # action = char_to_action.get(char, None)
        # if action is None:
        #     sys.exit()
        # event_happened = False
        action = np.array([0,0,0,0])
        for event in pygame.event.get():
            event_happened = True
            if event.type == QUIT:
                sys.exit()
            if event.type == KEYDOWN:
                char = event.dict['key']
                new_action = char_to_action.get(chr(char), None)
                if new_action is not None:
                    action = new_action
                else:
                    action = np.array([0 , 0 , 0 , 0])
                print("got char:", char)
                print("action", action)
        # pygame.event.pump()

        # action = env.action_space.sample()*10
        # delta = (env.get_block_pos() - env.get_endeff_pos())[:2]
        # action[:2] = delta * 100
        # action[1] -= 0.05
        # action = np.sign(action)
        # action += np.random.normal(size=action.shape) * 0.2
        # error = np.linalg.norm(delta)
        # print("action", action)
        # print("error", error)
        # if error < 0.04:
        #     action[1] += 10
        obs, reward, done, info = env.step(action)

        env.render()
        # print("action", action)
        if done:
            break
    print("new episode")



