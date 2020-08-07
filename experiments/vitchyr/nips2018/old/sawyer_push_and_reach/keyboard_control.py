import sys

from rlkit.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYEasyEnv
from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.policies.simple import ZeroPolicy
import numpy as np

print("making env")
# env = SawyerPushAndReachXYEasyEnv()
env = SawyerPushAndReachXYEnv()
env = MultitaskToFlatEnv(env)

policy = ZeroPolicy(env.action_space.low.size)
es = OUStrategy(
    env.action_space,
    theta=1
)
es = EpsilonGreedy(
    action_space=env.action_space,
    prob_random_action=0.1,
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
    'w': np.array([0 , -1, 0 , 0]),
    'a': np.array([1 , 0 , 0 , 0]),
    's': np.array([0 , 1 , 0 , 0]),
    'd': np.array([-1, 0 , 0 , 0]),
    'q': np.array([1 , -1 , 0 , 0]),
    'e': np.array([-1 , -1 , 0, 0]),
    'z': np.array([1 , 1 , 0 , 0]),
    'c': np.array([-1 , 1 , 0 , 0]),
    # 'm': np.array([1 , 1 , 0 , 0]),
    # 'j': np.array([.1 , 0 , 0 , 0]),
    # 'k': np.array([0 , .1 , 0 , 0]),
    'x': 'toggle',
    'r': 'reset',
}

ACTION_FROM = 'controller'
# ACTION_FROM = 'pd'
# ACTION_FROM = 'random'
H = 100000
# H = 300
# H = 50


lock_action = False
while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    action, _ = policy.get_action(None)
    for t in range(H):
        done = False
        if ACTION_FROM == 'controller':
            if not lock_action:
                action = np.array([0,0,0,0])
            for event in pygame.event.get():
                event_happened = True
                if event.type == QUIT:
                    sys.exit()
                if event.type == KEYDOWN:
                    char = event.dict['key']
                    new_action = char_to_action.get(chr(char), None)
                    if new_action == 'toggle':
                        lock_action = not lock_action
                    elif new_action == 'reset':
                        done = True
                    elif new_action is not None:
                        action = new_action
                    else:
                        action = np.array([0 , 0 , 0 , 0])
                    print("got char:", char)
                    print("action", action)
                    print("angles", env.data.qpos.copy())
        elif ACTION_FROM == 'random':
            action = env.action_space.sample()
        else:
            delta = (env.get_block_pos() - env.get_endeff_pos())[:2]
            action[:2] = delta * 100
        # if t == 0:
        #     print("goal is", env.get_goal_pos())
        # print("ee pos", env.get_endeff_pos())
        # action[1] -= 0.05
        # action = np.sign(action)
        # action += np.random.normal(size=action.shape) * 0.2
        # error = np.linalg.norm(delta)
        # print("action", action)
        # print("error", error)
        # if error < 0.04:
        #     action[1] += 10
        obs, reward, _, info = env.step(action)

        env.render()
        # print("action", action)
        if done:
            break
    print("new episode")



