import numpy as np
from rlkit.misc.asset_loader import get_relative_path
from rlkit.torch.sets.offline_rl_launcher import generate_trajectories


def main():
    # snapshot = 'manual-upload/disco-policy/snapshot_for_hand2xy_hand2x_1obj2xy_1obj2x.pkl'
    snapshot = 'manual-upload/disco-policy/snapshot_for_hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pkl'
    trajectories = generate_trajectories(
        snapshot_path=snapshot,
        max_path_length=20,
        num_steps=1e5,
        # num_steps=1000,
        save_observation_keys=['state_observation', 'image_observation'],
    )
    save_path = get_relative_path(
        'manual-upload/disco-policy/generated_100Ksteps_pathlen20_hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.npy'
    )
    np.save(save_path, trajectories)
    print("saved trajectories to", save_path)


if __name__ == '__main__':
    main()
