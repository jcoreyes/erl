"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.launchers.experiments.ashvin.hand_dapg import experiment, process_args

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule

if __name__ == "__main__":
    variant = dict(
        num_exps_per_instance=1,
        region='us-west-2',
        env_id='relocate-v0',
        sparse_reward=True,
        algorithm="NPG",
        rl_num_iter=1251,
    )

    search_space = {
        'seedid': range(3),
        'env_id': ['pen-v0', 'pen-notermination-v0', 'pen-sparse-v0',
                    'door-v0', 'door-sparse-v0',
                    'relocate-v0', 'relocate-sparse-v0',
                    'hammer-v0', 'hammer-sparse-v0', ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
