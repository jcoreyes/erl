from multiworld.core.image_env import ImageEnv
import multiworld.envs.mujoco as mwmj
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0

from rlkit.demos.collect_demo import collect_demos
import os.path as osp

from rlkit.misc.asset_loader import load_local_or_remote_file

if __name__ == '__main__':
    data = load_local_or_remote_file('11-16-door-reset-free-state-td3-sweep-params-policy-update-period/11-16-door_reset_free_state_td3_sweep_params_policy_update_period_2019_11_17_00_26_50_id000--s89728/params.pkl')
    env = data['evaluation/env']
    policy = data['trainer/trained_policy']
    presampled_goals_path = osp.join(
                osp.dirname(mwmj.__file__),
                "goals",
                "door_goals.npy",
            )
    presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
    image_env = ImageEnv(
                env,
                48,
                init_camera=sawyer_door_env_camera_v0,
                transpose=True,
                normalize=True,
                presampled_goals=presampled_goals,
    )
    collect_demos(image_env, policy, "data/local/demos/door_demos_action_noise_1000.npy", N=1000, horizon=100, threshold=.1, add_action_noise=True, key='angle_difference', render=False)
