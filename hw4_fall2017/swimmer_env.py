import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os.path as osp


MODEL_DIR = osp.abspath(osp.dirname(__file__))

class SwimmerEnvNew(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        full_path = osp.join(MODEL_DIR, 'swimmer.xml')
        mujoco_env.MujocoEnv.__init__(self, full_path, 10)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        ctrl_cost_coeff = 0.0001
        #xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        #xposafter = self.model.data.qpos[0, 0]
        #reward_fwd = (xposafter - xposbefore) / self.dt
        #reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()

        scaling = 50
        reward_fwd = self.get_body_comvel("torso")[0]
        reward_ctrl = -0.5 * (1e-2) * np.sum(np.square(a/scaling))
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        com = self.get_body_com("torso")
        return np.concatenate([qpos.flat, qvel.flat, com.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()