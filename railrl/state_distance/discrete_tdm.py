from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.policies.argmax import ArgmaxDiscretePolicy
from railrl.state_distance.exploration import MakeUniversal
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
        # self.policy = MakeUniversal(ArgmaxDiscretePolicy(qf))
        # exploration_policy = PolicyWrappedWithExplorationStrategy(
        #     exploration_strategy=exploration_strategy,
        #     policy=self.policy,
        # )
        DQN.__init__(self, env, qf, replay_buffer=replay_buffer, **dqn_kwargs,
                     **base_kwargs)
        self.policy = MakeUniversal(self.policy)
        self.eval_policy = MakeUniversal(self.eval_policy)
        self.exploration_policy = MakeUniversal(self.exploration_policy)
        self.eval_sampler = MultigoalSimplePathSampler(
            env=env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            discount_sampling_function=self._sample_max_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=False,
        )

    def _do_training(self):
        DQN._do_training(self)

    def evaluate(self, epoch):
        DQN.evaluate(self, epoch)
