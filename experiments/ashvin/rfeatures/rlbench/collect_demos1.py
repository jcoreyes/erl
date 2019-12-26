from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.tasks import FS10_V1
from rlbench.tasks.open_drawer import OpenDrawer
from rlbench.observation_config import ObservationConfig, CameraConfig
import numpy as np

import skvideo.io

import signal
import sys

live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

save_to_dir = 'gitignore/rlbench/demo_door_fixed1/'

camera_config = CameraConfig(image_size=(500, 300))
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.left_shoulder_camera = camera_config
# obs_config.right_shoulder_camera = camera_config
obs_config.set_all_low_dim(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(action_mode, DATASET, obs_config, headless=False, )
                # static_positions=True, )
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

demos = []

for j in range(10):
    demo = task.get_demos(1, live_demos=True)  # -> List[List[Observation]]
    demo = np.array(demo).flatten()
    demos.append(demo)

np.save(save_to_dir + "demos6.npy", demos)
