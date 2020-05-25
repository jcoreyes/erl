import pickle
import numpy as np

envs = [
    "door-binary-v0", "pen-binary-v0", "relocate-binary-v0",
    "door-sparse-v0", "pen-sparse-v0", "relocate-sparse-v0",
]

for env_name in envs:
    data = pickle.load(open("/home/ashvin/code/hand_dapg/dapg/demonstrations/railrl_%s_demos.pickle" % env_name, "rb"))
    y = []

    for path in data:
        observations = []
        actions = []
        infos = []
        rewards = []

        for t in range(len(path["observations"])):
            ob = path["observations"][t, :]
            action = np.clip(path["actions"][t, :], -1, 1)
            reward = path["rewards"][t]
            terminal = 0
            agent_info = {} # todo (need to unwrap each key)
            env_info = {} # todo (need to unwrap each key)

            o = dict()
            o["observation"] = ob
            o["state_observation"] = ob
            observations.append(o)
            actions.append(action)
            rewards.append(reward)
            infos.append({})

        H = len(observations) - 1

        traj = dict(
            observations=observations[:H],
            actions=actions[:H],
            rewards=np.array(rewards),
            next_observations=observations[1:H+1],
            terminals=np.zeros((H, )),
            agent_infos=infos[:H],
            env_infos=infos[:H],
        )

        y.append(traj)

    np.save("/home/ashvin/data/s3doodad/demos/icml2020/hand/sparsity/railrl_%s_demos.npy" % env_name, y)
