from collections import OrderedDict
import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import mujoco_py

from railrl.core import logger

from railrl.core.serializable import Serializable
from railrl.envs.env_utils import get_asset_full_path
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths


class SawyerXYZEnv(MujocoEnv, Serializable, MultitaskEnv):
    """Implements a 3D-position controlled Sawyer environment"""

    def __init__(self, reward_info=None, frame_skip=30):
        self.quick_init(locals())
        self.reward_info = reward_info
        MultitaskEnv.__init__(self, distance_metric_order=2)
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)


        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

        self.observation_space = Box(
            np.array([-0.2, 0.5, 0]),
            np.array([0.2, 0.7, 0.5]),
        )
        self.goal_space = Box(
            np.array([-0.2, 0.5, 0]),
            np.array([0.2, 0.7, 0.5]),
        )
        self.reset()
        self.reset_mocap_welds()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_gripper_mocap.xml')

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

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.mocap_set_action(a[:3] / 100, relative=True)
        self.set_goal_xyz(self.multitask_goal)
        u = np.zeros((8))
        u[7] = a[3]
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, u, obs, self.multitask_goal)
        done = False

        distance = np.linalg.norm(self.get_goal_pos() - self.get_endeff_pos())
        block_distance = np.linalg.norm(self.get_goal_pos() - self.get_block_pos())
        touch_distance = np.linalg.norm(self.get_endeff_pos() - self.get_block_pos())
        info = dict(
            distance=distance,
            block_distance=block_distance,
            touch_distance=touch_distance,
        )
        return obs, reward, done, info

    def _get_obs(self):
        p = self.get_endeff_pos()
        return p

    def get_block_pos(self):
        return self.data.body_xpos[self.block_id].copy()

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_goal_pos(self):
        return self.data.body_xpos[self.goal_id].copy()

    def sample_block_xyz(self):
        pos = np.random.uniform(
            np.array([-0.2, 0.5, 0.02]),
            np.array([0.2, 0.7, 0.02]),
        )
        return pos

    def sample_goal_for_rollout(self):
        return self.sample_goal_xyz()

    def sample_goal_xyz(self):
        pos = np.random.uniform(
            np.array([-0.2, 0.5, 0.0]),
            np.array([0.2, 0.7, 0.2]),
        )
        return pos

    def set_block_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[8:11] = pos.copy()
        qvel[8:11] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def set_goal_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[15:18] = pos.copy()
        qvel[15:18] = [0, 0, 0]
        self.set_state(qpos, qvel)

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
        # print("end eff at", np.array([self.data.body_xpos[self.endeff_id]]))
        # for d in self.data.body_xpos[self.endeff_id]:
        #     print(float(d))
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    def mocap_set_action(self, action, relative=True):
        pos_delta = action[None]

        if relative:
            self.reset_mocap2body_xpos()
            self.data.set_mocap_pos('mocap', self.data.mocap_pos + pos_delta)
        else:
            self.data.set_mocap_pos('mocap', pos_delta)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def reset(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.init_angles
        self.set_state(angles.flatten(), velocities.flatten())
        self.multitask_goal = self.sample_goal_xyz()
        self.set_goal_xyz(self.multitask_goal)
        self.set_block_xyz(self.sample_block_xyz())
        return self._get_obs()

    def compute_reward(self, ob, action, next_ob, goal):
        if not self.reward_info or self.reward_info["type"] == "euclidean":
            r = -np.linalg.norm(ob - goal)
        elif self.reward_info["type"] == "sparse":
            t = self.reward_info["threshold"]
            r = float(np.linalg.norm(ob - goal) < t) - 1
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def compute_her_reward_np(self, ob, action, next_ob, goal):
        if not self.reward_info or self.reward_info["type"] == "euclidean":
            r = -np.linalg.norm(next_ob - goal)
        elif self.reward_info["type"] == "sparse":
            t = self.reward_info["threshold"]
            r = float(np.linalg.norm(next_ob - goal) < t) - 1
        return r

    @property
    def init_angles(self):
        return [0.57242702304722737, -0.81917120114392261,
                1.0209690144401942, 1.0277836100084827, -0.62290997014344518,
                1.6426888531833115, 3.1387809209984603,
                0.0052920636104349323, -0.13972798601989481,
                0.5022168160162902, 0.020992940518284438,
                0.99998456929726953, 2.2910279298625033e-06,
                8.1234733355258378e-06, 0.0055552764211284642,
                -0.1230211958752539, 0.69090634842186527, -19.449133777272831, 1.0, 0.0, np.pi/4, 0.0]

    @property
    def goal_dim(self):
        return 3

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def block_id(self):
        return self.model.body_names.index('block')

    @property
    def goal_id(self):
        return self.model.body_names.index('goal')

    def convert_obs_to_goals(self, obs):
        return obs

    def sample_goals(self, batch_size):
        assert batch_size == 1
        return [self.multitask_goal]

    def log_diagnostics(self, paths, logger=logger, prefix=""):
        super().log_diagnostics(paths)

        statistics = OrderedDict()
        for stat_name in [
            'distance',
            'block_distance',
            'touch_distance',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s %s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s %s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)


class SawyerPickAndPlaceEnv(SawyerXYZEnv):
    def __init__(
            self,
            randomize_goals=True,
            only_reward_block_to_goal=False,
            **kwargs
    ):
        self.quick_init(locals())
        self.only_reward_block_to_goal = only_reward_block_to_goal
        self.randomize_goals = randomize_goals

        super().__init__(**kwargs)

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

        self.observation_space = Box(
            np.array([-0.2, 0.5, 0, -0.2, 0.5, 0]),
            np.array([0.2, 0.7, 0.5, 0.2, 0.7, 0.5]),
        )
        self.goal_space = Box(
            np.array([-0.2, 0.5, 0]),
            np.array([0.2, 0.7, 0.5]),
        )

    def sample_block_xyz(self):
        pos = np.random.uniform(
            np.array([0.0, 0.55, 0.02]),
            np.array([0.2, 0.7, 0.02]),
        )
        return pos

    def set_block_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[8:11] = pos.copy()
        qvel[8:11] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def compute_reward(self, ob, action, next_ob, goal):
        hand = self.get_endeff_pos()
        block = self.get_block_pos()
        block_goal = self.multitask_goal

        hand_to_block = np.linalg.norm(hand - block)
        block_to_goal = np.linalg.norm(block - block_goal)
        if self.only_reward_block_to_goal:
            return - block_to_goal
        else:
            return - hand_to_block - block_to_goal

    def compute_her_reward_np(self, ob, action, next_ob, goal):
        hand = next_ob[-6:-3]
        block = next_ob[-3:]
        block_goal = goal

        hand_to_block = np.linalg.norm(hand - block)
        block_to_goal = np.linalg.norm(block - block_goal)
        if self.only_reward_block_to_goal:
            return - block_to_goal
        else:
            return - hand_to_block - block_to_goal

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_block_pos()
        return np.concatenate((e, b))

    def sample_goal_xyz(self):
        if self.randomize_goals:
            pos = np.random.uniform(
                np.array([-0.2, 0.5, 0.02]),
                np.array([0.2, 0.7, 0.5]),
            )
        else:
            pos = np.array([0.0, 0.6, 0.02])
        return pos

    def convert_obs_to_goals(self, obs):
        return obs[:, -3:]


class SawyerPushEnv(SawyerPickAndPlaceEnv):
    """
    Take out the gripper action and fix the goal Z position to the table.
    Also start the block between the gripper and the goal
    """
    def __init__(self, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)

        self.action_space = Box(
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
        )

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.mocap_set_action(a[:3] / 100, relative=True)
        self.set_goal_xyz(self.multitask_goal)
        u = np.zeros((8))
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, u, obs, self.multitask_goal)
        done = False

        distance = np.linalg.norm(self.get_goal_pos() - self.get_endeff_pos())
        block_distance = np.linalg.norm(self.get_goal_pos() - self.get_block_pos())
        touch_distance = np.linalg.norm(self.get_endeff_pos() - self.get_block_pos())
        info = dict(
            distance=distance,
            block_distance=block_distance,
            touch_distance=touch_distance,
        )
        return obs, reward, done, info

    def sample_goal_xyz(self):
        if self.randomize_goals:
            pos = np.random.uniform(
                np.array([0.2, 0.5, 0.02]),
                np.array([-0.2, 0.7, 0.02]),
            )
        else:
            pos = np.array([0.2, 0.6, 0.02])
        return pos

    def sample_block_xyz(self):
        pos = np.random.uniform(
            np.array([-0.2, 0.5, 0.02]),
            np.array([0.2, 0.7, 0.02]),
        )
        while np.linalg.norm(pos[:2]) < 0.1:
            pos = np.random.uniform(
                np.array([-0.2, 0.5, 0.02]),
                np.array([0.2, 0.7, 0.02]),
            )
        return pos


class SawyerPushXYEnv(SawyerPushEnv):
    """
    Only move along XY-axis.
    """
    def __init__(self, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)

        self.action_space = Box(
            np.array([-1, -1]),
            np.array([1, 1]),
        )

    @property
    def init_angles(self):
        # return [6.49751287e-01, -6.13245189e-01, 3.68179034e-01,
        #         1.55969534e+00, -4.67787323e-01, 7.11615700e-01,
        #         2.97853855e+00, 3.12823177e-02, 1.82764768e-01,
        #         6.01241700e-01, 2.09921520e-02, 9.99984569e-01,
        #         1.29839525e-06,  4.60381956e-06,  5.55527642e-03,
        #         -1.09103281e-01, 6.55806890e-01,  7.29068781e-02,
        #         1, 0, 0, 0]
        return [
            1.02866769e+00, - 6.95207647e-01,   4.22932911e-01,
             1.76670458e+00, - 5.69637604e-01,   6.24117280e-01,
             3.53404635e+00,   2.99584816e-02, - 2.00417049e-02,
             6.07093769e-01,   2.10679106e-02,   9.99910945e-01,
             - 4.60349085e-05, - 1.78179392e-03, - 1.32259491e-02,
             1.07586388e-02, 6.62018003e-01,   2.09936716e-02,
             1.00000000e+00,   3.76632959e-14, 1.36837913e-11,
             1.56567415e-23
        ]


    def step(self, a):
        a = np.clip(a, -1, 1) / 100
        # mocap_delta_z = 0.02 - self.data.mocap_pos[0, 2]
        mocap_delta_z = 0
        new_mocap_action = np.hstack((a[:2], np.array([mocap_delta_z])))
        self.mocap_set_action(new_mocap_action, relative=True)
        self.set_goal_xyz(self.multitask_goal)
        u = np.zeros((8))
        u[7] = 1
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, u, obs, self.multitask_goal)
        done = False

        distance = np.linalg.norm(self.get_goal_pos() - self.get_endeff_pos())
        block_distance = np.linalg.norm(self.get_goal_pos() - self.get_block_pos())
        touch_distance = np.linalg.norm(self.get_endeff_pos() - self.get_block_pos())
        info = dict(
            distance=distance,
            block_distance=block_distance,
            touch_distance=touch_distance,
        )
        return obs, reward, done, info

if __name__ == "__main__":
    e = SawyerXYZEnv(reward_info=dict(type="euclidean"))
    for j in range(50):
        e.reset()
        for i in range(8):
            desired = np.random.uniform(
                low=(-0.01, -0.01, -0.01, -1),
                high=(0.01, 0.01, 0.01, 1)
            )
            print(i)
            for k in range(200):
                e.step(desired)

                e.render()
