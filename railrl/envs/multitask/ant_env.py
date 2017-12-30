from collections import OrderedDict

import numpy as np

from railrl.envs.mujoco.ant import AntEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.core.serializable import Serializable
from rllab.misc import logger as rllab_logger


class GoalXYPosAnt(AntEnv, MultitaskEnv, Serializable):
    def __init__(self, max_distance=2, use_low_gear_ratio=True):
        Serializable.quick_init(self, locals())
        self.max_distance = max_distance
        MultitaskEnv.__init__(self)
        super().__init__(use_low_gear_ratio=use_low_gear_ratio)
        self.set_goal(np.array([self.max_distance, self.max_distance]))

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goals(self, batch_size):
        return np.random.uniform(
            -self.max_distance,
            self.max_distance,
            (batch_size, 2),
        )

    def set_goal(self, goal):
        super().set_goal(goal)
        site_pos = self.model.site_pos.copy()
        site_pos[0, 0:2] = goal
        site_pos[0, 2] = 0.5
        self.model.site_pos = site_pos

    def convert_obs_to_goals(self, obs):
        return obs[:, 27:29]

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            self.get_body_com("torso"),
        ])

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
        xy_pos = self.convert_ob_to_goal(ob)
        pos_error = np.linalg.norm(xy_pos - self.multitask_goal)
        reward = - pos_error
        info_dict['x_pos'] = xy_pos[0]
        info_dict['y_pos'] = xy_pos[1]
        info_dict['dist_from_origin'] = np.linalg.norm(xy_pos)
        info_dict['desired_x_pos'] = self.multitask_goal[0]
        info_dict['desired_y_pos'] = self.multitask_goal[1]
        info_dict['desired_dist_from_origin'] = (
            np.linalg.norm(self.multitask_goal)
        )
        info_dict['pos_error'] = pos_error
        info_dict['goal'] = self.multitask_goal
        return ob, reward, done, info_dict

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths, logger=rllab_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)

        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('x_pos', 'X Position'),
            ('y_pos', 'Y Position'),
            ('dist_from_origin', 'Distance from Origin'),
            ('desired_x_pos', 'Desired X Position'),
            ('desired_y_pos', 'Desired Y Position'),
            ('desired_dist_from_origin', 'Desired Distance from Origin'),
            ('pos_error', 'Distance to goal'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stat,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for name_in_env_infos, name_to_log in [
            ('dist_from_origin', 'Distance from Origin'),
            ('desired_dist_from_origin', 'Desired Distance from Origin'),
            ('pos_error', 'Distance to goal'),
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

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)


class GoalXYPosAndVelAnt(AntEnv, MultitaskEnv, Serializable):
    def __init__(self, max_speed=0.05, max_distance=2, use_low_gear_ratio=True):
        Serializable.quick_init(self, locals())
        self.max_distance = max_distance
        self.max_speed = max_speed
        MultitaskEnv.__init__(self)
        super().__init__(use_low_gear_ratio=use_low_gear_ratio)
        self.set_goal(np.array([
            self.max_distance,
            self.max_distance,
            self.max_speed,
            self.max_speed,
        ]))

    @property
    def goal_dim(self) -> int:
        return 4

    def sample_goals(self, batch_size):
        return np.random.uniform(
            np.array([
                -self.max_distance,
                -self.max_distance,
                -self.max_speed,
                -self.max_speed
            ]),
            np.array([
                self.max_distance,
                self.max_distance,
                self.max_speed,
                self.max_speed
            ]),
            (batch_size, 4),
        )

    def convert_obs_to_goals(self, obs):
        return np.hstack((
            obs[:, 27:29],
            obs[:, 30:32],
        ))

    def set_goal(self, goal):
        super().set_goal(goal)
        site_pos = self.model.site_pos.copy()
        site_pos[0, 0:2] = goal[:2]
        site_pos[0, 2] = 0.5
        self.model.site_pos = site_pos

    def _get_obs(self):
        raise NotImplementedError()

    def _step(self, action):
        # get_body_comvel doesn't work, so you need to save the last position
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before

        # Idk what this is, but it's in the default ant, so I'll leave it in.
        state = self.state_vector()
        notdone = (
                np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        )
        done = not notdone

        ob = np.hstack((
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            self.get_body_com("torso"),
            torso_velocity,
        ))
        current_features = self.convert_ob_to_goal(ob)

        pos_error = np.linalg.norm(
            current_features[:2] - self.multitask_goal[:2]
        )
        vel_error = np.linalg.norm(
            current_features[2:] - self.multitask_goal[2:]
        )
        reward = - vel_error - pos_error
        info_dict = dict(
            goal=self.multitask_goal,
            vel_error=vel_error,
            pos_error=pos_error,
        )
        return ob, reward, done, info_dict

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return np.hstack((
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            self.get_body_com("torso"),
            np.zeros(3),  # init velocity is zero
        ))

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths, logger=rllab_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)
        vel_errors = get_stat_in_paths(
            paths, 'env_infos', 'vel_error'
        )
        pos_errors = get_stat_in_paths(
            paths, 'env_infos', 'pos_error'
        )

        statistics = OrderedDict()
        for stat, name in [
            (pos_errors, 'pos errors'),
            (vel_errors, 'vel errors'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)
