import numpy as np

from rllab.core.serializable import Serializable
from .mujoco_env import MujocoEnv
from softqlearning.misc import logger


class PusherEnv3DOF(MujocoEnv, Serializable):

    FILE = '3link_gripper_push_2d.xml'

    def __init__(self, goal=(0, -1), arm_distance_coeff=0):
        super(PusherEnv3DOF, self).__init__()
        Serializable.quick_init(self, locals())

        goal = np.array(goal)

        self._goal_mask = np.invert(np.isnan(goal))
        self._goal = goal[self._goal_mask]

        self._arm_distance_coeff = arm_distance_coeff
        self._action_cost_coeff = 0.1

    def step(self, a):
        reward, info = self.compute_reward(self.get_current_obs(), a, True)

        self.forward_dynamics(a)
        ob = self.get_current_obs()
        done = False

        return ob, reward, done, info

    def compute_reward(self, obss, actions, return_env_info):
        is_batch = False
        if obss.ndim == 1:
            obss = obss[None]
            actions = actions[None]
            is_batch = True

        arm_pos = obss[:, -6:-3]
        obj_pos = obss[:, -3:]
        obj_pos_masked = obj_pos[:, self._goal_mask]

        goal_dists = np.linalg.norm(self._goal[None] - obj_pos_masked, 1)
        arm_dists = np.linalg.norm(arm_pos - obj_pos, 1)
        ctrl_costs = np.sum(actions**2, 1)

        rewards = - self._action_cost_coeff * ctrl_costs - goal_dists
        rewards -= self._arm_distance_coeff * arm_dists

        if not is_batch:
            rewards = rewards.squeeze()
            arm_dists = arm_dists.squeeze()
            goal_dists = goal_dists.squeeze()

        if return_env_info:
            return rewards, dict(
                arm_distance=arm_dists,
                goal_distance=goal_dists
            )
        else:
            return rewards

    def viewer_setup(self):
        # self.itr = 0
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        rotation_angle = np.random.uniform(low=-0, high=360)
        if hasattr(self, "_kwargs") and 'vp' in self._kwargs:
            rotation_angle = self._kwargs['vp']
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset(self):

        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.squeeze()
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

        qacc = np.zeros(self.model.data.qacc.shape[0])
        ctrl = np.zeros(self.model.data.ctrl.shape[0])

        full_state = np.concatenate((qpos, qvel, qacc, ctrl))
        super(PusherEnv3DOF, self).reset(full_state)

        return self.get_current_obs()

    def get_current_obs(self):
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
