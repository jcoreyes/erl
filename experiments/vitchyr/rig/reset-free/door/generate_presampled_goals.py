from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from railrl.torch.vae.dataset.generate_goal_dataset import generate_goal_dataset_using_policy

env = SawyerDoorEnv(
    xml_path='sawyer_xyz/sawyer_door_pull_30.xml',
    min_angle=-0.523599,
)
image_env = ImageEnv(
    env,
    48,
    init_camera=sawyer_door_env_camera_v2,
    transpose=True,
    normalize=True,
    non_presampled_goal_img_is_garbage=True,
)

generate_goal_dataset_using_policy(
    env=image_env,
    num_goals=1000,
    use_cached_dataset=False,
    policy_file='manual-upload/SawyerDoorEnv_policy_params.pkl',
    path_length=30,
    show=True,
    save_file_prefix='/home/vitchyr/git/railrl/data/doodads3/manual-upload'
                  '/goals_n1000_SawyerDoorEnv_max_angle_30.npy'
)
