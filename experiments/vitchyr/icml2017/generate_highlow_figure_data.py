import random

from rlkit.envs.memory.high_low import HighLow
from rlkit.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from rlkit.launchers.memory_bptt_launchers import (
    trpo_launcher,
    mem_trpo_launcher,
    rtrpo_launcher,
    ddpg_launcher,
    mem_ddpg_launcher,
    rdpg_launcher,
    our_method_launcher_add_defaults,
)
from rlkit.misc.hyperparameter import DeterministicHyperparameterSweeper

if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "7-19-dev-generate_highlow_figure_data"

    n_seeds = 5
    mode = "ec2"
    exp_prefix = "7-19-generate_highlow_figure_data"

    env_class = HighLow

    use_gpu = True
    if mode != "here":
        use_gpu = False

    H = 25
    num_steps_per_iteration = 100
    num_steps_per_eval = 1000
    num_iterations = 100
    batch_size = 200
    memory_dim = 100
    # H = 5
    # num_steps_per_iteration = 10
    # num_steps_per_eval = 100
    # num_iterations = 10
    # batch_size = 5
    # memory_dim = 10
    # noinspection PyTypeChecker
    variant = dict(
        H=H,
        env_class=env_class,
        env_params=dict(
            horizon=H,
            give_time=False,
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
        ddpg_launcher,
        mem_ddpg_launcher,
        rdpg_launcher,
        our_method_launcher_add_defaults,
    ]:
        search_space = {
            # 'env_class': [WaterMaze1D, WaterMazeEasy1D, WaterMazeMemory1D],
        }
        sweeper = DeterministicHyperparameterSweeper(search_space,
                                                     default_parameters=variant)
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for _ in range(n_seeds):
                seed = random.randint(0, 9999999)
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
