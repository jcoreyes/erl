"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.launchers.experiments.ashvin.bear_launcher import experiment

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        launcher_kwargs=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),
        demo_data="demos/icml2020/hand/pen2_sparse.npy",
        off_policy_data="demos/icml2020/hand/pen_bc_sparse4.npy",
        eval_freq=1000,
        algorithm="BEAR",
        env_name="pen-v0",
        distance_type="MMD",
        algo_kwargs=dict(
            version=0,
            # lambda_=0,
            # threshold=0.05,
            mode="auto",
            num_samples_match=5,
            mmd_sigma=20.0,
            lagrange_thresh=10.0,
            use_kl=False,
            use_ensemble=False,
            kernel_type="gaussian",
        ),
        use_bootstrap=False,
        bootstrap_dim=4,
        delta_conf=0.1,
        use_ensemble_variance=True,
        use_data_policy=False,
        num_random=10,
        margin_threshold=10,
        max_timesteps=1e6,
    )

    search_space = {
        'seedid': range(3),
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
