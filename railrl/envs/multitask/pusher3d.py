from collections import OrderedDict

from railrl.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from railrl.core import logger as default_logger


class MultitaskPusher3DEnv(MujocoEnv, MultitaskEnv):
    GOAL_ZERO_POS = [-0.35, -0.35, -0.3230]  # from xml
    OBJ_ZERO_POS = [0.35, -0.35, -0.275]  # from xml
    goal_low = [-0.4, -0.4]
    goal_high = [0.4, 0.0]

    def __init__(self, reward_coefs=(0.5, 0.375, 0.125), norm_order=2):
        self.init_serialization(locals())
        self.reward_coefs = reward_coefs
        self.norm_order = norm_order
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(
            self,
            "pusher_3d.xml",
            5,
            automatically_set_obs_and_action_space=True,
        )

    def _step(self, a):
        obj_to_arm = self.get_body_com("object") - self.get_body_com("tips_arm")
        obj_to_goal = self.get_body_com("object") - self.get_body_com("goal")
        # Only care about x and y axis.
        obj_to_arm = obj_to_arm[:2]
        obj_to_goal = obj_to_goal[:2]
        obj_to_arm_dist = np.linalg.norm(obj_to_arm, ord=self.norm_order)
        obj_to_goal_dist = np.linalg.norm(obj_to_goal, ord=self.norm_order)
        control_magnitude = np.linalg.norm(a)

        forward_reward_vec = [obj_to_goal_dist, obj_to_arm_dist, control_magnitude]
        reward = -sum(
            [coef * r for (coef, r) in zip(self.reward_coefs, forward_reward_vec)]
        )

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False

        return ob, reward, done, dict(
            obj_to_arm_dist=obj_to_arm_dist,
            obj_to_goal_dist=obj_to_goal_dist,
            control_magnitude=control_magnitude,
        )

    def _get_obs(self):
        obs = np.concatenate([
            self.model.data.qpos.flat[:3],
            self.model.data.qvel.flat[:3],
            self.get_body_com("tips_arm")[:2],
            self.get_body_com("object")[:2],
        ])
        return obs

    def reset_model(self):
        qpos = self.init_qpos
        qpos[:] = 0
        qpos[-4:-2] += self.np_random.uniform(-0.05, 0.05, 2)
        # qpos represents the OFFSET and not the absolute position
        qpos[-2:] = self.multitask_goal - self.GOAL_ZERO_POS[:2]
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv)
        qvel[-4:] = 0

        self.set_state(qpos, qvel)

        return self._get_obs()

    def sample_goals(self, batch_size):
        return np.random.uniform(
            self.goal_low,
            self.goal_high,
            (batch_size, self.goal_dim)
        ) + self.GOAL_ZERO_POS[:2]

    def convert_obs_to_goals(self, obs):
        return obs[:, 8:10]

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    @property
    def goal_dim(self) -> int:
        return 2

    def log_diagnostics(self, paths, logger=default_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)

        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('obj_to_arm_dist', 'Distance to arm'),
            ('obj_to_goal_dist', 'Distance to goal'),
            ('control_magnitude', 'Control Magnitude'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stat,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for name_in_env_infos, name_to_log in [
            ('obj_to_arm_dist', 'Distance to arm'),
            ('obj_to_goal_dist', 'Distance to goal'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name_to_log),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)
