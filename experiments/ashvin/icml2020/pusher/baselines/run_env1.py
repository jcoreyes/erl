"""
AWR + SAC from demo experiment
"""

# from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
# from rlkit.launchers.experiments.ashvin.awr_sac_gcrl import experiment, process_args

# import rlkit.misc.hyperparameter as hyp
# from rlkit.launchers.arglauncher import run_variants

# from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy

import numpy as np
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_leap import SawyerPushAndReachXYEnv

env = SawyerPushAndReachXYEnv(
    hand_low=(-0.20, 0.50),
    hand_high=(0.20, 0.70),
    puck_low=(-0.20, 0.50),
    puck_high=(0.20, 0.70),
    goal_low=(-0.20, 0.50, -0.20, 0.50),
    goal_high=(0.20, 0.70, 0.20, 0.70),
    fix_reset=False,
    sample_realistic_goals=False,
    reward_type='hand_and_puck_distance',
    invisible_boundary_wall=True,
)

for i in range(10):
    env.reset()
    for t in range(100):
        env.step(-np.random.random((2, )))
        env.render()
