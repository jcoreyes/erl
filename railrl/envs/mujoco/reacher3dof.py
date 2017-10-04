import numpy as np

from railrl.envs.mujoco.tuomas_mujoco_env import TuomasMujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides

from rllab.core.serializable import Serializable



class ReacherEnv3DOF(TuomasMujocoEnv, Serializable):

    FILE = '3link_gripper_reach_2d.xml'

    def __init__(self):
        super(ReacherEnv3DOF, self).__init__()
        Serializable.quick_init(self, locals())
        # utils.EzPickle.__init__(self)
        # mujoco_env.MujocoEnv.__init__(self, '3link_gripper_reach_2d.xml', 5)

    def step(self, a):
        parm = self.get_body_com("distal_4")
        pgoal = self.get_body_com("goal")
        reward_dist = - np.linalg.norm(parm-pgoal)
        self.forward_dynamics(a)
        # self.do_simulation(a, self.frame_skip)
        ob = self.get_current_obs()
        done = False
        self.reward_orig = -reward_dist
        return ob, reward_dist, done, dict(dist=-reward_dist)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid=0
        self.viewer.cam.distance = 4.0
        rotation_angle = np.random.uniform(low=0, high=360)
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid=-1

    def reset(self):
        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + np.squeeze(self.init_qpos)
        self.goal = np.concatenate([np.random.uniform(low=-1.1, high=-0.5, size=1),
                 np.random.uniform(low=0.5, high=1.1, size=1)])
        qpos[-2:] = self.goal
        qvel = np.squeeze(self.init_qvel.copy())
        qvel[-4:] = 0
        qacc = np.zeros(self.model.data.qacc.shape[0])
        ctrl = np.zeros(self.model.data.ctrl.shape[0])
        full_state = np.concatenate((qpos, qvel, qacc, ctrl))
        super(ReacherEnv3DOF, self).reset(full_state)

        return self.get_current_obs()

    @overrides
    def get_current_obs(self):
            return np.concatenate([
                self.model.data.qpos.flat[:-4],
                self.model.data.qvel.flat[:-4],
                self.get_body_com("distal_4"),
                self.get_body_com("goal"),
            ])

    @overrides
    def log_diagnostics(self, paths):
        dists = [p['env_infos']['dist'][-1] for p in paths]

        logger.record_tabular('FinalDistanceAvg', np.mean(dists))
        logger.record_tabular('FinalDistanceMax', np.max(dists))
        logger.record_tabular('FinalDistanceMin', np.min(dists))
        logger.record_tabular('FinalDistanceStd', np.std(dists))

