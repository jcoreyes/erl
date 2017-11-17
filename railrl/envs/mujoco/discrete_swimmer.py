import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces
from rllab.misc import logger
import itertools


class DiscreteSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)

        self.action_range = 4
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)
        bounds = self.model.actuator_ctrlrange.copy()
        self.low = bounds[:, 0] / 5
        self.high = bounds[:, 1] / 5
        joint0_range = np.arange(self.low[0], self.high[0]+0.1, 0.5)
        joint1_range = np.arange(self.low[1], self.high[1] + 0.1, 0.5)
        joint0_range = np.array([-0.1, 0, 0.1])
        joint1_range = np.array([-0.1, 0, 0.1])
        self.continuous_actions = list(itertools.product(joint0_range, joint1_range))
        self.action_space = spaces.Discrete(len(self.continuous_actions))

    def _step(self, a):
        if not self.action_space or not self.action_space.contains(a):
            continuous_action = a
        else:
            continuous_action = self.continuous_actions[a]

        ctrl_cost_coeff = 0.0001
        # xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(continuous_action, self.frame_skip)
        # xposafter = self.model.data.qpos[0, 0]
        # reward_fwd = (xposafter - xposbefore) / self.dt
        reward_fwd = self.get_body_comvel("torso")[0]
        reward_ctrl = - ctrl_cost_coeff * np.square(continuous_action).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def log_diagnostics(self, paths):
        forward_rew = np.array([np.mean(traj['env_infos']['reward_fwd']) for traj in paths])
        reward_ctrl = np.array([np.mean(traj['env_infos']['reward_ctrl']) for traj in paths])

        logger.record_tabular('AvgRewardDist', np.mean(forward_rew))
        logger.record_tabular('AvgRewardCtrl', np.mean(reward_ctrl))


def main():
    env = DiscreteSwimmerEnv()
    for i in range(10000):
        env.step(env.action_space.sample())
        env.render()


if __name__ == '__main__':
    main()