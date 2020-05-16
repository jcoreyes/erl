"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.demos.source.mdp_path_loader import MDPPathLoader
from railrl.launchers.experiments.ashvin.hand_dapg import experiment, process_args

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.sac.policies import GaussianPolicy
from railrl.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule

if __name__ == "__main__":
    variant = dict(
        num_exps_per_instance=1,
        region='us-west-2',
        env_id='relocate-v0',
        sparse_reward=True,
    )

    search_space = {
        'seedid': range(3),
        'env_id': ['pen-v0', 'pen-notermination-v0', 'pen-sparse-v0', 'pen-binary-v0', 'door-v0',
                    'door-sparse-v0', 'door-binary-v0', 'relocate-v0', 'relocate-sparse-v0',
                    'relocate-binary-v0', 'hammer-v0', 'hammer-sparse-v0', 'hammer-binary-v0'],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
