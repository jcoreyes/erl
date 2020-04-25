import warnings
from typing import Any, Callable, List

import numpy as np
from gym.spaces import Box, Dict
from multiworld.core.multitask_env import MultitaskEnv

from railrl import pythonplusplus as ppp
from railrl.core.distribution import DictDistribution
from railrl.envs.contextual import ContextualRewardFn
from railrl.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
)
from railrl.envs.images import Renderer

import railrl.torch.pytorch_util as ptu

Goal = Any
GoalConditionedDiagnosticsFn = Callable[
    [List[Path], List[Goal]],
    Diagnostics,
]

class GoalDictDistributionFromMultitaskEnv(DictDistribution):
    def __init__(
            self,
            env: MultitaskEnv,
            desired_goal_keys=('desired_goal',),
    ):
        self._env = env
        self._desired_goal_keys = desired_goal_keys
        env_spaces = self._env.observation_space.spaces
        self._spaces = {
            k: env_spaces[k]
            for k in self._desired_goal_keys
        }

    def sample(self, batch_size: int):
        return {
            k: self._env.sample_goals(batch_size)[k]
            for k in self._desired_goal_keys
        }

    @property
    def spaces(self):
        return self._spaces


class AddImageDistribution(DictDistribution):
    def __init__(
            self,
            env: MultitaskEnv,
            base_distribution: DictDistribution,
            renderer: Renderer,
            image_goal_key='image_desired_goal',
            _suppress_warning=False,
    ):
        self._env = env
        self._base_distribution = base_distribution
        img_space = Box(0, 1, renderer.image_shape, dtype=np.float32)
        self._spaces = base_distribution.spaces
        self._spaces[image_goal_key] = img_space
        self._image_goal_key = image_goal_key
        self._renderer = renderer
        self._suppress_warning = _suppress_warning

    def sample(self, batch_size: int):
        if batch_size > 1 and not self._suppress_warning:
            warnings.warn(
                "Sampling many goals is slow. Consider using "
                "PresampledImageAndStateDistribution"
            )
        contexts = self._base_distribution.sample(batch_size)
        images = []
        for i in range(batch_size):
            goal = ppp.treemap(lambda x: x[i], contexts, atomic_type=np.ndarray)
            env_state = self._env.get_env_state()
            self._env.set_to_goal(goal)
            img_goal = self._renderer.create_image(self._env)
            self._env.set_env_state(env_state)
            images.append(img_goal)

        contexts[self._image_goal_key] = np.array(images)
        return contexts

    @property
    def spaces(self):
        return self._spaces


class AddLatentDistribution(DictDistribution):
    def __init__(
            self,
            dist,
            input_key,
            output_key,
            model,
    ):
        self.dist = dist
        self._spaces = dist.spaces
        self.input_key = input_key
        self.output_key = output_key
        self.model = model
        self.representation_size = self.model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[output_key] = latent_space

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size)
        s[self.output_key] = self.model.encode_np(s[self.input_key])
        return s

    @property
    def spaces(self):
        return self._spaces


class PriorDistribution(DictDistribution):
    def __init__(
            self,
            model,
            key,
    ):
        self._spaces = {}
        self.key = key
        if type(model) is str:
            self.model = load_local_or_remote_file(model)
        else:
            self.model = model
        self.representation_size = self.model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[key] = latent_space

    def sample(self, batch_size: int):
        mu, sigma = 0, 1 # sample from prior
        n = np.random.randn(batch_size, self.representation_size)
        s = {self.key: sigma * n + mu}
        return s

    @property
    def spaces(self):
        return self._spaces


class PresampledDistribution(DictDistribution):
    def __init__(
            self,
            slow_sampler: DictDistribution,
            num_presampled_goals,
    ):
        self._sampler = slow_sampler
        self._num_presampled_goals = num_presampled_goals
        self._presampled_goals = self._sampler.sample(num_presampled_goals)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._num_presampled_goals, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals

    @property
    def spaces(self):
        return self._sampler.spaces


class ContextualRewardFnFromMultitaskEnv(ContextualRewardFn):
    def __init__(
            self,
            env: MultitaskEnv,
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            observation_key='observation',
    ):
        self._env = env
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self._observation_key = observation_key

    def __call__(self, states, actions, next_states, contexts):
        del states
        obs = {
           self._achieved_goal_key: next_states[self._observation_key],
           self._desired_goal_key: contexts[self._desired_goal_key],
        }
        return self._env.compute_rewards(actions, obs)


class GoalConditionedDiagnosticsToContextualDiagnostics(ContextualDiagnosticsFn):
    # use a class rather than function for serialization
    def __init__(
            self,
            goal_conditioned_diagnostics: GoalConditionedDiagnosticsFn,
            goal_key: str
    ):
        self._goal_conditioned_diagnostics = goal_conditioned_diagnostics
        self._goal_key = goal_key

    def __call__(self, paths: List[Path], contexts: List[Context]) -> Diagnostics:
        goals = [c[self._goal_key] for c in contexts]
        return self._goal_conditioned_diagnostics(paths, goals)
