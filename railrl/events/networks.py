import torch
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
import numpy as np


class BetaQ(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            **flatten_mlp_kwargs
    ):
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=(
                    self.observation_dim + self.action_dim + self.goal_dim + 1
            ),
            output_size=self.goal_dim if vectorized else 1,
            hidden_activation=torch.tanh,
            output_activation=torch.sigmoid,
            **flatten_mlp_kwargs
        )
        self.env = env
        self.vectorized = vectorized

    def forward(self, observations, actions, goals, num_steps_left, **kwargs):
        flat_inputs = torch.cat(
            (observations, actions, goals, num_steps_left),
            dim=1,
        )
        return super().forward(flat_inputs, **kwargs)

    def create_eval_function(self, obs, goal, num_steps_left):
        def beta_eval(a1, a2):
            actions = np.array([[a1, a2]])
            return self.eval_np(
                observations=np.array([[
                    *obs
                ]]),
                actions=actions,
                goals=np.array([[
                    *goal
                ]]),
                num_steps_left=np.array([[num_steps_left]])
            )[0, 0]
        return beta_eval


class BetaV(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            **flatten_mlp_kwargs
    ):
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=(
                    self.observation_dim + self.goal_dim + 1
            ),
            output_size=self.goal_dim if vectorized else 1,
            hidden_activation=torch.tanh,
            output_activation=torch.sigmoid,
            **flatten_mlp_kwargs
        )
        self.env = env
        self.vectorized = vectorized

    def forward(self, observations, goals, num_steps_left, **kwargs):
        flat_inputs = torch.cat(
            (observations, goals, num_steps_left),
            dim=1,
        )
        return super().forward(flat_inputs, **kwargs)


class TanhFlattenMlpPolicy(TanhMlpPolicy):
    def __init__(
            self,
            env,
            **kwargs
    ):
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        self.env = env
        super().__init__(
            input_size=self.observation_dim + self.goal_dim + 1,
            output_size=self.action_dim,
            **kwargs
        )

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
            **kwargs
    ):
        flat_input = torch.cat((observations, goals, num_steps_left), dim=1)
        return super().forward(flat_input, **kwargs)

    def get_action(self, ob_np, goal_np, tau_np):
        actions = self.eval_np(
            ob_np[None],
            goal_np[None],
            tau_np[None],
        )
        return actions[0, :], {}
