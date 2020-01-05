from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.tasks import FS10_V1
from rlbench.tasks.open_drawer import OpenDrawer
from rlbench.observation_config import ObservationConfig, CameraConfig
import numpy as np

import skvideo.io

live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

save_to_dir = 'gitignore/rbench/'

camera_config = CameraConfig(image_size=(500, 300))
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.left_shoulder_camera = camera_config
obs_config.right_shoulder_camera = camera_config
obs_config.set_all_low_dim(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(action_mode, DATASET, obs_config, False)
env.launch()

# train_tasks = FS10_V1['train']
# test_tasks = FS10_V1['test']
# task_to_train = np.random.choice(train_tasks, 1)[0]
# import ipdb; ipdb.set_trace()

# print(action_mode.action_size)

task = env.get_task(OpenDrawer)
task.sample_variation()  # random variation
descriptions, obs = task.reset()
# obs, reward, terminate = task.step(np.random.normal(size=action_mode.action_size))

# import ipdb; ipdb.set_trace()

task._robot

for j in range(3, 10):
    demos = task.get_demos(1, live_demos=True)  # -> List[List[Observation]]
    demos = np.array(demos).flatten()

    np.save(save_to_dir + "demos_%d.npy" % j, demos)

    d = demos

    obs_right = []
    obs_left = []
    for i in range(len(d)):
        obs_left.append(d[i].left_shoulder_rgb)
        obs_right.append(d[i].right_shoulder_rgb)

    videodata = (np.array(obs_left) * 255).astype(int)
    filename = save_to_dir + "demo_left_%d.mp4" % j
    skvideo.io.vwrite(filename, videodata)

    videodata = (np.array(obs_right) * 255).astype(int)
    filename = save_to_dir + "demo_right_%d.mp4" % j
    skvideo.io.vwrite(filename, videodata)
