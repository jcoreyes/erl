import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from rlkit.envs.erl.modify_xml import create_new_xml

class SparseSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, mods, frameskip=4):
        new_path = create_new_xml('swimmer.xml', mods)
        mujoco_env.MujocoEnv.__init__(self, new_path, frameskip)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_ctrl
        if xposafter > 1.0:
            reward += reward_fwd
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
