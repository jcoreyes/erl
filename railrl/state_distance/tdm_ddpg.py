from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.algos.ddpg import DDPG


class TdmDdpg(TemporalDifferenceModel, DDPG):
    def __init__(
            self,
            env,
            qf,
            exploration_policy,
            ddpg_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
    ):
        DDPG.__init__(
            self,
            env=env,
            qf=qf,
            policy=policy,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            **ddpg_kwargs,
            **base_kwargs
        )
        super().__init__(**tdm_kwargs)

    def _do_training(self):
        DDPG._do_training(self)

    def evaluate(self, epoch):
        DDPG.evaluate(self, epoch)
