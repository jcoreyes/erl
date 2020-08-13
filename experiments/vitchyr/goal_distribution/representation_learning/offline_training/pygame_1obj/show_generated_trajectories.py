import numpy as np
from rlkit.misc.asset_loader import get_relative_path
import cv2


def main():
    saved_path = get_relative_path(
        # 'manual-upload/disco-policy/generated_10_trajectories_hand2xy_hand2x_1obj2xy_1obj2x.npy'
        'manual-upload/disco-policy/generated_10_trajectories_hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.npy'
    )
    trajectories = np.load(saved_path, allow_pickle=True)
    print("number of trajectories = ", len(trajectories))
    for traj in trajectories:
        for img in traj['image_observation']:
            img = img.transpose()
            cv2.imshow('image', img)
            cv2.waitKey(10)


if __name__ == '__main__':
    main()
