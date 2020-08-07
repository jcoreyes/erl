import os
import joblib

import numpy as np

from rlkit.samplers.util import rollout

files = dict(
    reach_left=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-left/09-14_pusher-3dof-reacher-naf-yolo_left_2017_09_14_17_52_45_0010/params.pkl'
    ),
    reach_right=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-right/09-14_pusher-3dof-reacher-naf-yolo_right_2017_09_14_17_52_45_0016/params.pkl'
    ),
    reach_middle=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-middle/09-14_pusher-3dof-reacher-naf-yolo_middle_2017_09_14_17_52_45_0013/params.pkl'
    ),
    reach_bottom=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom/09-14_pusher-3dof-reacher-naf-yolo_bottom_2017_09_14_17_52_45_0019/params.pkl'
    ),
    merge_bottom_left=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-1-combine-naf-policies-left/09-14_1-combine-naf-policies-left_2017_09_14_21_42_24_0000--s-68077/params.pkl'
    ),
    merge_bottom_right=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-1-combine-naf-policies-right/09-14_1-combine-naf-policies-right_2017_09_14_21_42_29_0000--s-42677/params.pkl'
    ),
    merge_bottom_middle=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-1-combine-naf-policies-middle/09-14_1-combine-naf-policies-middle_2017_09_14_21_42_27_0000--s-91696/params.pkl'
    ),
    reach_bottom_left=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom-left/09-14_pusher-3dof-reacher-naf-yolo_bottom-left_2017_09_14_17_52_45_0001/params.pkl'
    ),
    reach_bottom_right=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom-right/09-14_pusher-3dof-reacher-naf-yolo_bottom-right_2017_09_14_17_52_45_0007/params.pkl'
    ),
    reach_bottom_middle=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom-middle/09-14_pusher-3dof-reacher-naf-yolo_bottom-middle_2017_09_14_17_52_45_0005/params.pkl'
    ),
)


for name, full_path in files.items():
    name = name.replace('_', '-')  # in case Tuomas's script cares

    data = joblib.load(full_path)
    if 'policy' in data:
        policy = data['policy']
    else:
        policy = data['naf_policy']
    env = data['env']

    print(name)

    pos_lst = list()
    for i in range(100):
        path = rollout(env, policy, max_path_length=300, animated=False)
        pos_lst.append(path['final_observation'][-3:-1])

    pos_all = np.stack(pos_lst)

    outfile = os.path.join('data/papers/icra2018/results/pusher/naf',
                           name + '.txt')
    np.savetxt(outfile, pos_all)
