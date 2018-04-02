from railrl.envs.ros.sawyer_reaching import SawyerJointSpaceReachingEnv
import numpy as np

env = SawyerJointSpaceReachingEnv(reward='huber', safety_force_magnitude=5, temperature=15, safety_box=True, use_safety_checks=False)
env.reset()
