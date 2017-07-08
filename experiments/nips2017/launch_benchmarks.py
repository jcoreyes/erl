from railrl.envs.memory.high_low import HighLow
from railrl.envs.pygame.water_maze import (
    WaterMaze,
    WaterMazeMemory,
    WaterMazeEasy,
    WaterMazeEasy1D,
    WaterMaze1D,
    WaterMazeMemory1D,
)
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from railrl.launchers.memory_bptt_launchers import (
    trpo_launcher,
    mem_trpo_launcher,
    rtrpo_launcher,
    ddpg_launcher,
    mem_ddpg_launcher,
    rdpg_launcher,
)
from railrl.misc.hyperparameter import DeterministicHyperparameterSweeper
from rllab.envs.mujoco.walker2d_env import Walker2DEnv

if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "7-7-dev-launch-benchmark"

    # n_seeds = 5
    # mode = "ec2"
    # exp_prefix = "fig1-6-15-ddpg-trpo-hl-h25-correct"

    # env_class = HighLow
    # env_class = WaterMazeMemory
    # env_class = WaterMaze
    env_class = WaterMazeEasy
    # env_class = WaterMazeEasy1D
    # env_class = WaterMazeMemory1D
    # env_class = WaterMaze1D
    # env_class = Walker2DEnv

    use_gpu = True
    if mode != "here":
        use_gpu = False

    H = 25
    num_steps_per_iteration = 100
    num_steps_per_eval = 1000
    num_iterations = 100
    batch_size = 200
    memory_dim = 100
    # noinspection PyTypeChecker
    variant = dict(
        H=H,
        env_class=env_class,
        env_params=dict(
            horizon=H,
        ),
        exp_prefix=exp_prefix,
        num_steps_per_iteration=num_steps_per_iteration,
        num_steps_per_eval=num_steps_per_eval,
        num_iterations=num_iterations,
        memory_dim=memory_dim,
        use_gpu=use_gpu,
        batch_size=batch_size,  # For DDPG only
    )
    exp_id = -1
    for launcher in [
        # trpo_launcher,
        # mem_trpo_launcher,
        # rtrpo_launcher,
        # ddpg_launcher,
        # mem_ddpg_launcher,
        rdpg_launcher,
    ]:
        search_space = {
            # 'env_class': [WaterMaze1D, WaterMazeEasy1D, WaterMazeMemory1D],
        }
        sweeper = DeterministicHyperparameterSweeper(search_space,
                                                     default_parameters=variant)
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for seed in range(n_seeds):
                exp_id += 1
                set_seed(seed)
                variant['seed'] = seed
                variant['exp_id'] = exp_id

                run_experiment(
                    launcher,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    snapshot_mode='last',
                    use_gpu=use_gpu,
                )
