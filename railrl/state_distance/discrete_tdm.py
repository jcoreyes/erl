from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.algos.dqn import DQN


class DiscreteTDM(TemporalDifferenceModel, DQN):
    def __init__(
            self,
            env,
            qf,
            dqn_kwargs,
            **kwargs
    ):
        super().__init__(env, qf, **kwargs)
        DQN.__init__(self, env, qf, **dqn_kwargs)

    def _do_training(self):
        DQN._do_training(self)

    def evaluate(self, epoch):
        DQN.evaluate(self, epoch)
