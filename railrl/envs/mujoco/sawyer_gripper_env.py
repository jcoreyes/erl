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

    def __init__(self, reward_info=None):
        self.quick_init(locals())
        self.reward_info = reward_info
        MultitaskEnv.__init__(self, distance_metric_order=2)
        MujocoEnv.__init__(self, self.model_name, frame_skip=10)


        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

        self.observation_space = Box(
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

    def sample_goal_xyz(self):
        pos = np.random.uniform(
            np.array([-0.2, 0.5, 0.0]),
            np.array([0.2, 0.7, 0.2]),
        )
        return pos

    def set_block_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        # pdb.set_trace()
        qpos[8:11] = pos # TODO: don't hardcode this
        # qpos[14] = np.random.uniform(0.0, np.pi) # randomize the orientation
        qvel[8:11] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def set_goal_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[15:18] = pos
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
        angles[:] = [0.57242702304722737, -0.81917120114392261, 1.0209690144401942, 1.0277836100084827, -0.62290997014344518, 1.6426888531833115, 3.1387809209984603, 0.0052920636104349323, -0.13972798601989481, 0.5022168160162902, 0.020992940518284438, 0.99998456929726953, 2.2910279298625033e-06, 8.1234733355258378e-06, 0.0055552764211284642, -0.1230211958752539, 0.69090634842186527, -19.449133777272831, 1.0, 0.0, 0.0, 0.0]
        self.set_state(angles.flatten(), velocities.flatten())
        self.multitask_goal = self.sample_goal_xyz()
        self.set_goal_xyz(self.multitask_goal)
        self.set_block_xyz(self.sample_block_xyz())
        return self._get_obs()

    def compute_reward(self, ob, action, next_ob, goal, info={}):
        if not self.reward_info or self.reward_info["type"] == "euclidean":
            r = -np.linalg.norm(ob - goal)
        elif self.reward_info["type"] == "sparse":
            t = self.reward_info["threshold"]
            r = float(np.linalg.norm(ob - goal) < t) - 1
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def multitask_reward(self, ob, action, next_ob, goal, info={}):
        if not self.reward_info or self.reward_info["type"] == "euclidean":
            r = -np.linalg.norm(next_ob - goal)
        elif self.reward_info["type"] == "sparse":
            t = self.reward_info["threshold"]
            r = float(np.linalg.norm(next_ob - goal) < t) - 1
        return r

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
        """Not implemented, since this is task specific"""
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

class SawyerBlockEnv(SawyerXYZEnv):
    """Block pushing env if goals_on_table, otherwise pick-and-place"""
    def __init__(self, goals_on_table=False, randomize_goals=True, randomize_goal_orientation=False):
        Serializable.quick_init(self, locals())
        self.viewer = None
        self.goals_on_table = goals_on_table
        self.randomize_goals = randomize_goals
        self.randomize_goal_orientation = randomize_goal_orientation

        super().__init__()

        self._action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        # self._action_space = Box(
        #     np.array([-0.01, -0.01, -0.01, -1]),
        #     np.array([0.01, 0.01, 0.01, 1]),
        # )

        self._observation_space = Box(
            np.array([-0.2, 0.5, 0, -0.2, 0.5, 0]),
            np.array([0.2, 0.7, 0.5, 0.2, 0.7, 0.5]),
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
        # pdb.set_trace()
        qpos[8:11] = pos # TODO: don't hardcode this
        if self.randomize_goal_orientation:
            qpos[14] = np.random.uniform(0.0, np.pi)
        qvel[8:11] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def compute_reward(self, ob, action, next_ob, goal, info={}):
        hand = self.get_endeff_pos()
        block = self.get_block_pos()
        block_goal = self.multitask_goal

        hand_to_block = np.linalg.norm(hand - block)
        block_to_goal = np.linalg.norm(block - block_goal)
        return -hand_to_block - block_to_goal

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_block_pos()
        return np.concatenate((e, b))

    def sample_goal_xyz(self):
        if self.randomize_goals:
            pos = np.random.uniform(
                np.array([-0.2, 0.5, 0.02]),
                np.array([0.2, 0.7, 0.02]),
            )
        else:
            pos = np.array([-0.0, 0.6, 0.02])
        return pos

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
