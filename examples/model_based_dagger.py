import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.dagger.controller import MPCController
from railrl.dagger.dagger import Dagger
from railrl.dagger.model import DynamicsModel
from railrl.envs.multitask.ant_env import GoalXYPosAnt
from railrl.envs.multitask.multitask_env import MultitaskEnvToSilentMultitaskEnv
from railrl.launchers.launcher_util import setup_logger
from railrl.torch.data_management.normalizer import TorchFixedNormalizer


def experiment(variant):
    # You can have any environment, but the main thing is that your environment
    # needs to implement `cost_fn(self, states, action, next_states`.
    #
    # See multitask_env.MultitaskEnv to see how it's implemented for this
    # environment.
    env = GoalXYPosAnt(max_distance=2)
    env = MultitaskEnvToSilentMultitaskEnv(env)

    observation_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    obs_normalizer = TorchFixedNormalizer(observation_dim)
    action_normalizer = TorchFixedNormalizer(action_dim)
    delta_normalizer = TorchFixedNormalizer(observation_dim)
    model = DynamicsModel(
        observation_dim=observation_dim,
        action_dim=action_dim,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
        delta_normalizer=delta_normalizer,
        **variant['model_kwargs']
    )
    mpc_controller = MPCController(
        env,
        model,
        env.cost_fn,
        **variant['mpc_controller_kwargs']
    )
    algo = Dagger(
        env,
        model,
        mpc_controller,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
        delta_normalizer=delta_normalizer,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            collection_mode='online',
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            num_updates_per_epoch=10,
            max_path_length=100,
            learning_rate=1e-3,
            num_updates_per_env_step=1,
            batch_size=128,
            num_paths_for_normalization=20,
        ),
        mpc_controller_kwargs=dict(
            num_simulated_paths=512,
            mpc_horizon=15,
        ),
        model_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
    )
    setup_logger("test-experiment")
    experiment(variant)
