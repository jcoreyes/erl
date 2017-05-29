from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml
from rllab.envs.env_spec import EnvSpec
from rllab.spaces import Box


class MujocoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, local_xml_file_name, frame_skip=1):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            get_asset_xml(local_xml_file_name),
            frame_skip=frame_skip,
        )
        self.observation_space = Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
        )
        self.action_space = Box(
            low=self.action_space.low,
            high=self.action_space.high,
        )
        self.spec = EnvSpec(
            self.observation_space,
            self.action_space,
        )

    def log_diagnostics(self, paths):
        pass
