import numpy as np

save_to_dir = 'gitignore/rlbench/demo_door_fixed1/'

x = np.load(save_to_dir + "demos5b.npy")

extract_keys = [
    "gripper_joint_positions",
    "gripper_open",
    "gripper_pose",
    "gripper_touch_forces",
    "joint_forces",
    "joint_positions",
    "joint_velocities",
    "task_low_dim_state",
    "reward",
]

y = []

for demo in x:
    observations = []
    actions = []
    infos = []
    rewards = []
    for obs in demo:
        o = dict()
        for key in extract_keys:
            o[key] = obs.__dict__[key]
        s1 = o["task_low_dim_state"]
        s2 = o["gripper_pose"]
        s3 = o["joint_positions"]
        # s = np.concatenate((s1, s2, s3))
        s = np.concatenate((s2, s3))
        o["state_observation"] = s
        o["state_achieved_goal"] = np.zeros((0, ))
        o["state_desired_goal"] = np.zeros((0, ))
        img = np.uint8(255 * obs.left_shoulder_rgb).transpose().flatten()
        o["image_observation"] = img
        observations.append(o)
        rewards.append(float(o["reward"]))
        infos.append({})

        u = np.zeros((8, ))
        u[:7] = o["joint_velocities"]
        u[7] = o["gripper_open"]
        actions.append(u)

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

np.save(save_to_dir + "demos5b_10_dict_joints.npy", y)
