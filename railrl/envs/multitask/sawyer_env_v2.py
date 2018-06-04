from railrl.core.serializable import Serializable
from railrl.envs.multitask.multitask_env import MultitaskEnv
from sawyer_control.sawyer_reaching import SawyerJointSpaceReachingEnv
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
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

class MultiTaskSawyerJointSpaceReachingEnv(MultitaskEnv, SawyerJointSpaceReachingEnv):
    def __init__(self, kwargs, distance_metric_order=2, goal_dim_weights=None):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(
            self,
            distance_metric_order=distance_metric_order,
            goal_dim_weights=goal_dim_weights,
        )
        SawyerJointSpaceReachingEnv.__init__(self, **kwargs)
    def set_goal(self, goal):
        self.multitask_goal = goal
        self.desired = goal

    @property
    def goal_dim(self):
        return 7

    def sample_goals(self, batch_size):
        return np.random.uniform(JOINT_ANGLES_LOW, JOINT_ANGLES_HIGH, size=(batch_size, 7))

    def convert_obs_to_goals(self, obs):
        return obs[:, :7]

class MultiTaskSawyerXYZReachingEnv(MultitaskEnv, SawyerXYZReachingEnv,Serializable):
    def __init__(self, kwargs, distance_metric_order=2, goal_dim_weights=None):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(
            self,
            distance_metric_order=distance_metric_order,
            goal_dim_weights=goal_dim_weights,
        )
        SawyerXYZReachingEnv.__init__(self, **kwargs)

    def set_goal(self, goal):
        self.multitask_goal = goal
        self.desired = goal

    @property
    def goal_dim(self):
        return 3

    def sample_goals(self, batch_size):
        rand = self._get_random_ee_pose(batch_size=batch_size)
        return rand

    def convert_obs_to_goals(self, obs):
        return obs[:, 21:24]