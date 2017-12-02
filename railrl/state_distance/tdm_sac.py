from railrl.sac.sac import SoftActorCritic
from railrl.state_distance.tdm import TemporalDifferenceModel


class TdmSac(TemporalDifferenceModel, SoftActorCritic):
    def __init__(
            self,
            env,
            qf,
            vf,
            sac_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
    ):
        SoftActorCritic.__init__(
            self,
            env=env,
            policy=policy,
            qf=qf,
            vf=vf,
            replay_buffer=replay_buffer,
            **sac_kwargs,
            **base_kwargs
        )
        super().__init__(**tdm_kwargs)

    def _do_training(self):
        SoftActorCritic._do_training(self)

    def evaluate(self, epoch):
        SoftActorCritic.evaluate(self, epoch)
