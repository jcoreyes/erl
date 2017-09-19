"""
Experiment with NAF.
"""
import random
from railrl.envs.pygame.water_maze import WaterMazeEasy1D, WaterMazeEasy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment, set_seed
from railrl.torch.naf import NAF, NafPolicy
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
import railrl.torch.pytorch_util as ptu


def example(variant):
    env_class = variant['env_class']
    seed = variant['seed']
    env_params = variant['env_params']
    algo_params = variant['algo_params']

    set_seed(seed)
    env = env_class(**env_params)
    es = OUStrategy(action_space=env.action_space)
    qf = NafPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
    )
    algorithm = NAF(
        env,
        naf_policy=qf,
        exploration_strategy=es,
        **algo_params
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    use_gpu = True
    horizon = 1000
    # noinspection PyTypeChecker
    variant = dict(
        env_class=HalfCheetahEnv,
        env_params=dict(
            # horizon=horizon,
        ),
        algo_params=dict(
            num_epochs=50,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=1024,
            replay_buffer_size=50000,
            max_path_length=horizon,
        ),
        version="Normal",
    )
    seed = random.randint(0, 10000)
    run_experiment(
        example,
        exp_prefix="naf-dev",
        seed=seed,
        mode='here',
        variant=variant,
        use_gpu=use_gpu,
    )
