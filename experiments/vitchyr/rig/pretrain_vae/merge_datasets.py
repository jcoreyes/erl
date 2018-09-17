import numpy as np

imgs1 = np.load(
    "/tmp/door_open_N100_sawyer_door_env_camera_imsize48_oracleFalse.npy"
)
imgs2 = np.load(
    "/home/vitchyr/git/railrl/data/doodads3/manual-upload/skewed_dataset_SawyerDoorEnv_N1000_sawyer_door_env_camera_imsize48_oracleFalse.npy"
)
imgs = np.concatenate((imgs1, imgs2), axis=0)
import ipdb; ipdb.set_trace()
np.save(
    "/home/vitchyr/git/railrl/data/doodads3/manual-upload/full_skewed_dataset_SawyerDoorEnv_N1000_sawyer_door_env_camera_imsize48_oracleFalse.npy",
    imgs
)
