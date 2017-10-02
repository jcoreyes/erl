import random

from railrl.algos.state_distance.state_distance_q_learning import \
    multitask_rollout
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.state_distance import PerfectPoint2DQF
from railrl.policies.state_distance import ArgmaxQFPolicy
from rllab.misc import logger

def experiment(variant):
    num_rollouts = variant['num_rollouts']
    H = variant['H']
    render = variant['render']
    env = MultitaskPoint2DEnv()
    qf = PerfectPoint2DQF()
    policy = variant['policy_class'](
        qf,
        env,
        **variant['policy_params']
    )
    paths = []
    for _ in range(num_rollouts):
        goal = env.sample_goal_state_for_rollout()
        path = multitask_rollout(
            env,
            policy,
            goal,
            discount=0,
            max_path_length=H,
            animated=render,
        )
        paths.append(path)
    env.log_diagnostics(paths)
    logger.dump_tabular(with_timestamp=False)

if __name__ == '__main__':
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev"
    version = "Dev"
    run_mode = "none"

    variant = dict(
        num_rollouts=10,
        H=300,
        render=False,
        policy_class=ArgmaxQFPolicy,
        policy_params=dict(
            sample_size=100,
            num_gradient_steps=0,
            sample_actions_from_grid=False,
        )
    )
    for exp_id in range(n_seeds):
        seed = random.randint(0, 999999)
        run_experiment(
            experiment,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
        )
