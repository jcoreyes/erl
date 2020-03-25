from multiworld.core.multitask_env import MultitaskEnv

from railrl.core.distribution import Distribution
from railrl.envs.contextual import ContextualRewardFn


class GoalDistributionFromMultitaskEnv(Distribution):
    def __init__(
            self,
            env: MultitaskEnv,
            desired_goal_key='desired_goal',
    ):
        self._env = env
        self._desired_goal_key = desired_goal_key

    def sample(self, batch_size: int):
        return self._env.sample_goals(batch_size)[self._desired_goal_key]

    @property
    def space(self):
        return self._env.observation_space.spaces[self._desired_goal_key]


class ContextualRewardFnFromMultitaskEnv(ContextualRewardFn):
    def __init__(
            self,
            env: MultitaskEnv,
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal_key',
    ):
        self._env = env
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key

    def __call__(self, states, actions, next_states, contexts):
        del states
        obs = {
           self._achieved_goal_key: next_states,
           self._desired_goal_key: contexts,
        }
        return self._env.compute_rewards(actions, obs)