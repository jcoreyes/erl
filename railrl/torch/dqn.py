import torch.optim as optim
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm


class DQN(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            exploration_policy,
            qf,
            learning_rate=1e-3,
            **kwargs
    ):
        super().__init__(env, exploration_policy, **kwargs)
        self.qf = qf
        self.target_qf = self.qf.copy()
        self.learning_rate = learning_rate
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )

        self.eval_statistics = None

    def training_mode(self, mode):
        self.qf.train(mode)
        self.target_qf.train(mode)

    def _do_training(self):
        batch = self.get_batch(training=True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Compute loss
        """

        next_actions = self.target_policy(next_obs)
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self.target_qf(next_obs).detach().max(1)
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions)
        bellman_errors = (y_pred - y_target) ** 2
        raw_qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Take the gradient step
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Save some statistics for eval
        """

    def cuda(self):
        self.qf.cuda()
        self.target_qf.cuda()

    def evaluate(self, epoch):
        pass

    def offline_evaluate(self, epoch):
        raise NotImplementedError()
