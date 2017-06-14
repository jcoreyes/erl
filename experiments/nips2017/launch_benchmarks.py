from railrl.envs.pygame.water_maze import WaterMaze
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

if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "6-14-dev-mbptt-ddpg-benchmarks-many-3"

    env_class = WaterMaze
    H = 25

    # noinspection PyTypeChecker
    variant = dict(
        H=H,
        env_class=env_class,
        env_params=dict(
            horizon=H,
        ),
        exp_prefix=exp_prefix,
        num_steps_per_iteration=100,
        num_steps_per_eval=100,
        num_iterations=10,
        memory_dim=30,
        use_gpu=True,
        batch_size=50,  # For DDPG only
    )
    exp_id = -1
    for launcher in [
        # rtrpo_launcher,
        # trpo_launcher,
        # mem_trpo_launcher,
        # mem_ddpg_launcher,
        # ddpg_launcher,
        rdpg_launcher,
    ]:
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
            )
