import numpy as np

from railrl.envs.mujoco.mujoco_env import MujocoEnv

class SawyerEnv(MujocoEnv):
    def __init__(self):
        self.init_serialization(locals())
        super().__init__('sawyer.xml')
        self.desired = np.zeros(7)
        
    #needs to return the observation, reward, done, and info
    def _step(self, a):
        obs = self._get_obs()
        reward = 0
        done = false
        info = {}
        return obs, reward, done, info
    def reset_model(self):
        pass
    def _get_obs(self):
        pass
    def viewer_setup(self):
        pass
#how does this environment command actions?
#how do we get observations? and what form do they take?
#how do we reset the model?
