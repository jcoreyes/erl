import numpy as np
import mujoco_py


class SawyerMocapBase(object):
    """
    Provides some commonly-shared functions.
    """
    MOCAP_LOW = [-0.1, 0.5, 0]
    MOCAP_HIGH = [0.1, 0.7, 0.5]

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # robot view
        # rotation_angle = 90
        # cam_dist = 1
        # cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

        # 3rd person view
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        # top down view
        # cam_dist = 0.2
        # rotation_angle = 0
        # cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    def get_block_pos(self):
        return self.data.body_xpos[self.block_id].copy()

    def get_puck_pos(self):
        return self.data.body_xpos[self.puck_id].copy()

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_goal_pos(self):
        return self.data.body_xpos[self.goal_id].copy()

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def puck_id(self):
        return self.model.body_names.index('puck')

    @property
    def block_id(self):
        return self.model.body_names.index('block')

    @property
    def goal_id(self):
        return self.model.body_names.index('goal')

    def mocap_set_action(self, action):
        pos_delta = action[None]
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.MOCAP_LOW,
            self.MOCAP_HIGH,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
