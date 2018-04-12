from sawyer_control.sawyer_reaching import SawyerJointSpaceReachingEnv
import numpy as np

JOINT_ANGLES_HIGH = np.array([
    1.70167993,
    1.04700017,
    3.0541791,
    2.61797006,
    3.05900002,
    2.09400001,
    3.05899961
])

JOINT_ANGLES_LOW = np.array([
    -1.70167995,
    -2.14700025,
    -3.0541801,
    -0.04995198,
    -3.05900015,
    -1.5708003,
    -3.05899989
])

class MultiTaskSawyerJointSpaceReachingEnv(SawyerJointSpaceReachingEnv):
    @property
    def goal_dim(self):
        return 7

    def sample_goals(self, batch_size):
        return np.random.uniform(JOINT_ANGLES_LOW, JOINT_ANGLES_HIGH, size=(batch_size, 7))[0]

    def convert_obs_to_goals(self, obs):
        return obs[:7] #DOUBLE CHECK, BUT I THINK THIS IS CORRECT!