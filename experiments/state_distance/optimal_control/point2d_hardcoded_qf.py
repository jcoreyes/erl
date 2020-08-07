import argparse
import random

from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv, PerfectPoint2DQF
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.policies import (
    ArgmaxQFPolicy,
    PseudoModelBasedPolicy,
    ConstrainedOptimizationOCPolicy,
)
from state_distance.rollout_util import multitask_rollout
from rlkit.core import logger


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='path to the snapshot file with a QF')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev"
    run_mode = 'none'
    use_gpu = True

    variant = dict(
        num_rollouts=1,
        H=50,
        render=True,
        policy_class=ConstrainedOptimizationOCPolicy,
        policy_params=dict(
            solver_params=dict(
                disp=False,
                maxiter=10,
            )
        ),
    )
    if run_mode == 'none':
        for exp_id in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                use_gpu=use_gpu,
            )
    elif run_mode == 'custom':
        for (policy_class, policy_params) in [
            (
                PseudoModelBasedPolicy,
                dict(
                    sample_size=1,
                    num_gradient_steps=100,
                )
            ),
            (
                PseudoModelBasedPolicy,
                dict(
                    sample_size=100,
                    num_gradient_steps=1,
                )
            ),
            (
                ArgmaxQFPolicy,
                dict(
                    sample_size=1,
                    num_gradient_steps=100,
                )
            ),
            (
                ArgmaxQFPolicy,
                dict(
                    sample_size=100,
                    num_gradient_steps=1,
                )
            ),
        ]:
            variant['policy_class'] = policy_class
            variant['policy_params'] = policy_params
            for exp_id in range(n_seeds):
                seed = random.randint(0, 999999)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                )
