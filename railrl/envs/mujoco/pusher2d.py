import abc

import numpy as np

from railrl.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger


class Pusher2DEnv(MujocoEnv, metaclass=abc.ABCMeta):

    FILE = '3link_gripper_push_2d.xml'

    def __init__(self, goal=(0, -1)):
        self.init_serialization(locals())
        if not isinstance(goal, np.ndarray):
            goal = np.array(goal)
        self._goal = goal
        super().__init__(
            '3link_gripper_push_2d.xml',
            automatically_set_obs_and_action_space=True,
        )

    def _step(self, a):
        vec = self.get_body_com("distal_4") - self.get_body_com("object")
        distance = np.linalg.norm(vec)
        reward = - distance
        goal_distance = np.linalg.norm(
            self.get_body_com("object")[:2] - self._goal
        )

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            distance=distance,
            arm_distance=distance,
            goal_distance=goal_distance,
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        rotation_angle = 90
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_model(self):
        qpos = (
            np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos.squeeze()
        )
        qpos[-3:] = self.init_qpos.squeeze()[-3:]
        while True:
            object_ = [np.random.uniform(low=-1.0, high=-0.4),
                       np.random.uniform(low=0.3, high=1.0)]
            # x and y are flipped
            goal = np.array([-1, 1])  # Not actual goal
            if np.linalg.norm(np.array(object_)-np.array(goal)) > 0.45:
                break
        self.object = np.array(object_)

        qpos[-4:-2] = self.object
        qvel = self.init_qvel.copy().squeeze()
        qvel[-4:] = 0


        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:-4],
            self.model.data.qvel.flat[:-4],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
        ])

    def log_diagnostics(self, paths):
        arm_dists = [p['env_infos']['arm_distance'][-1] for p in paths]
        goal_dists = [p['env_infos']['goal_distance'][-1] for p in paths]

        logger.record_tabular('FinalArmDistanceAvg', np.mean(arm_dists))
        logger.record_tabular('FinalArmDistanceMax',  np.max(arm_dists))
        logger.record_tabular('FinalArmDistanceMin',  np.min(arm_dists))
        logger.record_tabular('FinalArmDistanceStd',  np.std(arm_dists))

        logger.record_tabular('FinalGoalDistanceAvg', np.mean(goal_dists))
        logger.record_tabular('FinalGoalDistanceMax',  np.max(goal_dists))
        logger.record_tabular('FinalGoalDistanceMin',  np.min(goal_dists))
        logger.record_tabular('FinalGoalDistanceStd',  np.std(goal_dists))
