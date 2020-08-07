from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
import numpy as np
from rlkit.demos.collect_demo import collect_demos, collect_demos_fixed
from rlkit.misc.asset_loader import load_local_or_remote_file

from rlkit.envs.simple.point import Point

class PointmassExpert:
    def __init__(self, n):
        self.n = n

    def get_action(self, o):
        x = o[:self.n]
        g = o[self.n:]
        return np.clip(g - x, -1, 1) * np.random.random((self.n, )), {}

if __name__ == '__main__':
    env = Point(n=10, fixed_goal=(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, ))
    # data = load_local_or_remote_file('ashvin/icml2020/point/sac-online2/run10/id1/itr_100.pkl')
    # env = data['evaluation/env']
    # policy = data['evaluation/policy']
    # import ipdb; ipdb.set_trace()
    # policy =
    # policy.to("cpu")
    # image_env = ImageEnv(
    #     env,
    #     48,
    #     init_camera=sawyer_init_camera_zoomed_in,
    #     transpose=True,
    #     normalize=True,
    # )
    # env_name = pendulum
    policy = PointmassExpert(10)

    outfile = "/home/ashvin/data/s3doodad/demos/icml2020/point/demos_noisy_expert_fixed_goal1000.npy"
    horizon = 100
    collect_demos_fixed(env, policy, outfile, N=1000, horizon=horizon) # , threshold=.1, add_action_noise=False, key='puck_distance', render=True, noise_sigma=0.0)
    # data = load_local_or_remote_file("demos/pusher_demos_1000.npy")
    # for i in range(100):
    #     goal = data[i]['observations'][49]['desired_goal']
    #     o = env.reset()
    #     path_length = 0
    #     while path_length < 50:
    #         env.set_goal({'state_desired_goal':goal})
    #         o = o['state_observation']
    #         new_obs = np.hstack((o, goal))
    #         a, agent_info = policy.get_action(new_obs)
    #         o, r, d, env_info = env.step(a)
    #         path_length += 1
    #     print(i, env_info['puck_distance'])
