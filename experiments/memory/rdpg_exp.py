"""
RDPG Experiments
"""
import random

from rlkit.envs.memory.high_low import HighLow
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.memory_states.policies import RecurrentPolicy
from rlkit.memory_states.qfunctions import RecurrentQFunction
from rlkit.torch.rdpg import Rdpg


def example(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    es = OUStrategy(env_spec=env.spec)
    qf = RecurrentQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        10,
    )
    policy = RecurrentPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        10,
    )
    algorithm = Rdpg(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    use_gpu = True
    H = 32
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=100,
            num_steps_per_eval=100,
            batch_size=H*4,
            max_path_length=100,
            use_gpu=use_gpu,
        ),
        env_params=dict(
            num_steps=H,
            # horizon=H,
            # use_small_maze=True,
            # l2_action_penalty_weight=0,
        ),
        env_class=HighLow,
        # env_class=WaterMazeEasy,
    )
    seed = random.randint(0, 9999)
    run_experiment(
        example,
        exp_prefix="dev-pytorch-rdpg",
        # exp_prefix="dev-6-12-rdpg-small-water-maze-easy",
        seed=seed,
        mode='here',
        variant=variant,
        use_gpu=use_gpu,
    )
