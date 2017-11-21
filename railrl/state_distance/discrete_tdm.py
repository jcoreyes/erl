from railrl.misc.ml_util import ConstantSchedule
from railrl.torch.dqn import DQN


class DiscreteTDM(DQN):
    def __init__(
            self,
            env,
            qf,
            max_tau=10,
            epoch_max_tau_schedule=None,
            **kwargs
    ):
        """

        :param env:
        :param qf:
        :param epoch_max_tau_schedule: A schedule for the maximum planning
        horizon tau.
        :param kwargs:
        """
        super().__init__(env, qf, learning_rate=1e-3, use_hard_updates=False,
                         hard_update_period=1000, tau=0.001, epsilon=0.1,
                         **kwargs)
        self.max_tau = max_tau
        if epoch_max_tau_schedule is None:
            epoch_max_tau_schedule = ConstantSchedule(self.max_tau)
        self.epoch_max_tau_schedule = epoch_max_tau_schedule

    def _start_epoch(self, epoch):
        self.max_tau = self.epoch_max_tau_schedule.get_value(epoch)
        super()._start_epoch(epoch)

    def get_batch(self, training=True):
        batch = super().get_batch(training=training)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()
