import numpy as np
import time
import os.path as osp
from rlkit.misc.asset_loader import local_path_from_s3_or_local_path

def generate_vae_dataset(
        env_class,
        N=10000,
        test_p=0.9,
        use_cached=True,
        observation_key='observation',
        init_camera=None,
        dataset_path=None,
        env_kwargs=None,
        oracle_dataset=False,
        n_random_steps=100,
):
    if env_kwargs is None:
        env_kwargs = {}
    filename = "/tmp/{}_{}_{}_oracle{}.npy".format(
        env_class.__name__,
        str(N),
        init_camera.__name__ if init_camera else '',
        oracle_dataset,
    )
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
        N = dataset.shape[0]
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        now = time.time()
        env = env_class(**env_kwargs)
        env.reset()
        info['env'] = env
        observation_dim = env.observation_space.spaces[observation_key].low.size
        dataset = np.zeros((N, observation_dim))
        for i in range(N):
            if oracle_dataset:
                goal = env.sample_goal()
                env.set_to_goal(goal)
            else:
                env.reset()
                for _ in range(n_random_steps):
                    env.step(env.action_space.sample())[0]
            obs = env.step(env.action_space.sample())[0][observation_key]
            dataset[i, :] = obs
            print(i)
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    np.random.shuffle(dataset)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info