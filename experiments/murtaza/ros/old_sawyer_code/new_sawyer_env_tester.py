from railrl.envs.ros.sawyer_reaching import SawyerJointSpaceReachingEnv
import numpy as np
import pdb

def send_zeroes(env, length):
    zeroes = np.zeros(7)
    for _ in range(length):
        env.step(zeroes)

def send_one_hot(env, idx, length, strength=1):
    vec = np.zeros(7)
    vec[idx] = strength
    for _ in range(length):
        env.step(vec)



env = SawyerJointSpaceReachingEnv(reward='huber', safety_force_magnitude=5, temperature=15, safety_box=True, use_safety_checks=False)
env.reset()

