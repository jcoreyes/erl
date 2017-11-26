from railrl.state_distance.rollout_util import MultigoalSimplePathSampler
from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.algos.dqn import DQN


class DiscreteTDM(TemporalDifferenceModel, DQN):
    def __init__(
            self,
            env,
            qf,
            dqn_kwargs,
            tdm_kwargs,
            base_kwargs,
            replay_buffer=None,
    ):
        super().__init__(env, qf, **tdm_kwargs)
        DQN.__init__(self, env, qf, replay_buffer=replay_buffer, **dqn_kwargs,
                     **base_kwargs)

    def _do_training(self):
        DQN._do_training(self)

    def evaluate(self, epoch):
        DQN.evaluate(self, epoch)
