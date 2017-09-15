import numpy as np

from railrl.envs.mujoco.tuomas_mujoco_env import TuomasMujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger

JNT_INDS = np.array([0, 1, 2])
OB1_INDS = np.array([3, 4])
OB2_INDS = np.array([5, 6])
TGT_INDS = np.array([7, 8])

DIMS = 9


class PusherAvoiderEnv3DOF(TuomasMujocoEnv, Serializable):

    FILE = '3link_gripper_push_avoid_2d.xml'

    def __init__(self, task='avoid', hit_penalty=1.0, action_cost_coeff=0.1):
        Serializable.quick_init(self, locals())
        self._task = task

        self._action_cost_coeff = action_cost_coeff
        self._hit_penalty = hit_penalty

        super(PusherAvoiderEnv3DOF, self).__init__()

    def step(self, a):
        reward, info = self.compute_reward(self.get_current_obs(), a, True)

        self.forward_dynamics(a)
        ob = self.get_current_obs()
        done = False

        return ob, reward, done, info

    def compute_reward(self, observations, actions, return_env_info=False):
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]

        pos = observations[:, :DIMS]
        vel = observations[:, DIMS:]

        ob1_pos = pos[:, OB1_INDS]
        tgt_pos = pos[:, TGT_INDS]
        ob2_vel = vel[:, OB2_INDS]

        obj2_speed = np.linalg.norm(ob2_vel, axis=1)
        tgt_dist = np.linalg.norm(ob1_pos - tgt_pos, axis=1)

        rewards = 0
        if self._task == 'both' or self._task == 'avoid':
            if obj2_speed > 1E-3:
                rewards -= self._hit_penalty

        if self._task == 'both' or self._task == 'push':
            rewards -= tgt_dist

        rewards -= self._action_cost_coeff * np.sum(actions**2, axis=1)

        if return_env_info:
            return rewards, dict(
                goal_distance=tgt_dist,
                obj2_speed=obj2_speed
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

    def _check_collisions(self, objs):
        for i, obj in enumerate(objs):
            # Too close or too far.
            if not (0.4 < np.linalg.norm(obj) < 2):
                return True

            # Touching the arm.
            if obj[0] > 0 and np.abs(obj[1]) < 0.4:
                return True

            for j in range(i+1, len(objs)):
                obj2 = objs[j]
                if np.linalg.norm(obj-obj2) < 0.2:
                    return True

        return False

    def reset(self):
        qpos = np.zeros(DIMS)
        qvel = np.zeros(DIMS)
        qacc = np.zeros(DIMS)
        ctrl = np.zeros(DIMS)

        while True:
            obj_pos_all = (
                np.stack((  # obj1
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                )),
                np.stack((  # obj2 want to avoid
                    np.random.uniform(0.2, 1.5),
                    np.random.uniform(-0.5, 0.5),
                )),
                np.stack((  # target
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                ))
            )
            # obj_pos_all = np.random.uniform(-1, 1, (3, 2))

            if not self._check_collisions(obj_pos_all):
                break

        qpos[OB1_INDS] = obj_pos_all[0]
        qpos[OB2_INDS] = obj_pos_all[1]
        qpos[TGT_INDS] = obj_pos_all[2]

        state = np.concatenate((qpos, qvel, qacc, ctrl))

        super(PusherAvoiderEnv3DOF, self).reset(state)

        return self.get_current_obs()

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com('distal_4')
        ])

    def log_diagnostics(self, paths):
        obj2_speed = np.concatenate([p['env_infos']['obj2_speed'] for p in paths])
        goal_dists = [p['env_infos']['goal_distance'][-1] for p in paths]

        logger.record_tabular('obj2Speed', np.mean(obj2_speed))

        logger.record_tabular('FinalGoalDistanceAvg', np.mean(goal_dists))
        logger.record_tabular('FinalGoalDistanceMax',  np.max(goal_dists))
        logger.record_tabular('FinalGoalDistanceMin',  np.min(goal_dists))
        logger.record_tabular('FinalGoalDistanceStd',  np.std(goal_dists))
