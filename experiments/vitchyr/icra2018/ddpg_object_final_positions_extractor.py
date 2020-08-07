import os
import joblib

import numpy as np

from rlkit.samplers.util import rollout

bottom_path = (
    '/home/vitchyr/git/rllab-rail/railrl/data/papers/icra2017/'
    '09-11_pusher-3dof-horizontal-2_2017_09_11_23_23_50_0039/'
    'itr_50.pkl'
)
vertical_path = dict(
    middle=(
        '/home/vitchyr/git/rllab-rail/railrl/data/papers/icra2017/'
        '09-11_pusher-3dof-vertical-2_2017_09_11_23_24_08_0017/'
        'itr_50.pkl'
    ),
    left=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/'
        '09-12-pusher-3dof-vertical-l2-left/'
        '09-12_pusher-3dof-vertical-l2-left_2017_09_12_15_56_43_0001/'
        'itr_40.pkl'
    ),
    right=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/'
        '09-12-pusher-3dof-vertical-l2-right/'
        '09-12_pusher-3dof-vertical-l2-right_2017_09_12_15_57_16_0001/'
        'itr_40.pkl'
    ),
)
merge_path = dict(
    middle=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/'
        '09-14-combine-policies--middle-bottom/'
        '09-14_combine-policies--middle-bottom_2017_09_14_14_39_01_0000--s-2893/'
        'params.pkl'
    ),
    left=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/'
        '09-14-combine-policies--left-bottom/'
        '09-14_combine-policies--left-bottom_2017_09_14_14_39_36_0000--s-4401/'
        'params.pkl'
    ),
    right=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/'
        '09-14-combine-policies--right-bottom/'
        '09-14_combine-policies--right-bottom_2017_09_14_14_38_47_0000--s-984/'
        'params.pkl'
    ),
)

files = (
    # (vertical_path['left'], 'reach-left'),
    # (vertical_path['middle'], 'reach-middle'),
    # (vertical_path['right'], 'reach-right'),
    # (bottom_path, 'reach-bottom'),
    (
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/'
         '09-14-pusher-3dof-reacher-bottom-middle/'
         '09-14_pusher-3dof-reacher-bottom-middle_2017_09_14_17_13_07_0001/'
         'params.pkl',
         'reach-bottom-middle'
    ),
    (
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/'
        '09-14-pusher-3dof-reacher-bottom-right/'
        '09-14_pusher-3dof-reacher-bottom-right_2017_09_14_17_13_22_0001/'
        'params.pkl',
        'reach-bottom-right'
    ),
    (
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/'
        '09-14-pusher-3dof-reacher-bottom-left/'
        '09-14_pusher-3dof-reacher-bottom-left_2017_09_14_17_12_41_0001/'
        'params.pkl',
        'reach-bottom-left'
    ),

    # (merge_path['left'], 'merge-bottom-left'),
    # (merge_path['middle'], 'merge-bottom-middle'),
    # (merge_path['right'], 'merge-bottom-right'),
)


for full_path, name in files:

    data = joblib.load(full_path)
    policy = data['policy']
    env = data['env']

    print(name)

    pos_lst = list()
    for i in range(100):
        path = rollout(env, policy, max_path_length=300, animated=False)
        pos_lst.append(path['final_observation'][-3:-1])

    pos_all = np.stack(pos_lst)

    outfile = os.path.join('data/papers/icra2018/results/pusher/ddpg',
                           name + '.txt')
    np.savetxt(outfile, pos_all)
