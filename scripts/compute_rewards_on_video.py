# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv

import sys
from multiworld.core.image_env import ImageEnv
from multiworld.envs.real_world.sawyer.sawyer_reaching import SawyerReachXYZEnv
# from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv

from railrl.launchers.launcher_util import run_experiment
# import railrl.util.hyperparameter as hyp
from railrl.launchers.experiments.ashvin.rfeatures.encoder_wrapped_env import EncoderWrappedEnv
from railrl.misc.asset_loader import load_local_or_remote_file

import torch

from railrl.launchers.experiments.ashvin.rfeatures.rfeatures_model import TimestepPredictionModel
import numpy as np

from railrl.torch.grill.video_gen import VideoSaveFunction

from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp

import railrl.torch.pytorch_util as ptu

# from railrl.launchers.experiments.ashvin.rfeatures.rfeatures_trainer import TimePredictionTrainer

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torchvision.utils import save_image
import pickle

demo_trajectory_rewards = []

import torchvision
from PIL import Image
import torchvision.transforms.functional as TF
import random

import skvideo.io

RANDOM_CROP_X = 16
RANDOM_CROP_Y = 16
WIDTH = 456
HEIGHT = 256
CROP_WIDTH = WIDTH - RANDOM_CROP_X
CROP_HEIGHT = HEIGHT - RANDOM_CROP_Y

eps = 1e-5

t_to_pil = torchvision.transforms.ToPILImage()
t_random_resize = torchvision.transforms.RandomResizedCrop(
    size=(CROP_WIDTH, CROP_HEIGHT,),
    scale=(0.9, 1.0),
    ratio=(1.0, 1.0), # don't change aspect ratio
)
t_color_jitter = torchvision.transforms.ColorJitter(
    brightness=0.2, # (0.8, 1.2),
    contrast=0.2, # (0.8, 1.2),
    saturation=0.2, # (0.8, 1.2),
    hue=0.1, # (-0.2, 0.2),
)
t_to_tensor = torchvision.transforms.ToTensor()

def get_random_crop_params(img, scale_x, scale_y):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (PIL Image): Image to be cropped.
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    w = int(random.uniform(*scale_x) * CROP_WIDTH)
    h = int(random.uniform(*scale_y) * CROP_HEIGHT)

    i = random.randint(0, img.size[1] - h)
    j = random.randint(0, img.size[0] - w)
    
    return i, j, h, w

def load_path(data, demo_path):
    H = len(data)
    print("loading video, size", data.shape)

    rewards = []

    # import ipdb; ipdb.set_trace()

    t_color_jitter_instance = None

    # obs_batch = data[:, 60:, :440, :] # right
    obs_batch = data[:, 60:, 60:, :] # left
    # np.zeros((num_obs, 240, 440, 3))
    obs_batch = obs_batch.transpose(0, 3, 1, 2) / 255.0
    zs = env._encode(obs_batch)

    z0 = zs[0, :]
    zT = zs[-1, :]

    for reward_type in ["latent_distance", "regression_distance"]:

        for i in range(len(zs)):
            dt = zs[i, :] - z0
            dT = zT - z0

            if reward_type == "regression_distance":
                regression_pred_yt = (dt * dT).sum() / ((dT ** 2).sum() + eps)
                r = -np.abs(1-regression_pred_yt)
            if reward_type == "latent_distance":
                r = -np.linalg.norm(dt - dT)

            rewards.append(r)

        reward_filename = demo_path[:-4] + "_%s_imagenet.png" % reward_type
        plt.figure(figsize=(8, 8))
        plt.plot(rewards)
        plt.savefig(reward_filename)

    # # Order of these two lines matters
    # env.zT = goal_image_transformed
    # env.initialize(zs)
    # # print("z0", env.z0, "zT", env.zT, "dT", env.dT)

    # for i in range(H):
    #     ob = path[i]
    #     action = path["actions"][i]
    #     reward = path["rewards"][i]
    #     next_ob = path["next_observations"][i]
    #     terminal = path["terminals"][i]
    #     agent_info = path["agent_infos"][i]
    #     env_info = path["env_infos"][i]

    #     # goal = path["goal"]["state_desired_goal"][0, :]
    #     # import ipdb; ipdb.set_trace()
    #     # print(goal.shape, ob["state_observation"])
    #     # state_observation = np.concatenate((ob["state_observation"], goal))
    #     # action = action[:2]

    #     # update_obs_with_latent(ob)
    #     # update_obs_with_latent(next_ob)
    #     env._update_obs_latent(ob, zs[i, :])
    #     env._update_obs_latent(next_ob, zs[i+1, :])
    #     reward = env.compute_reward(
    #         action,
    #         next_ob,
    #     )
    #     path["rewards"][i] = reward
    #     # reward = np.array([reward])
    #     # terminal = np.array([terminal])

    #     # print(reward)
    #     rewards.append(reward)
    # demo_trajectory_rewards.append(rewards)

def load_demos(demo_paths, processed_demo_path, reference_path, name):
    datas = []
    for demo_path in demo_paths:
        data = skvideo.io.vread(demo_path)
        load_path(data, demo_path)
        print("Finished loading demo: " + demo_path)

    # np.save(processed_demo_path, data)
    print("Dumping data")
    # pickle.dump(datas, open(processed_demo_path, "wb"), protocol=4)

    plt.figure(figsize=(8, 8))
    print("Demo trajectory rewards len: ", len(demo_trajectory_rewards), "Data len: ", len(datas))
    pickle.dump(demo_trajectory_rewards, open("demos/rlbench/demo_rewards_%s.p" % name, "wb"), protocol=4)
    for r in demo_trajectory_rewards:
        plt.plot(r)
    plt.savefig("demos/rlbench/demo_rewards_%s.png" %name)

def update_obs_with_latent(obs):
    latent_obs = env._encode_one(obs["image_observation"])
    latent_goal = np.zeros([]) # env._encode_one(obs["image_desired_goal"])
    obs['latent_observation'] = latent_obs
    obs['latent_achieved_goal'] = latent_goal
    obs['latent_desired_goal'] = latent_goal
    obs['observation'] = latent_obs
    obs['achieved_goal'] = latent_goal
    obs['desired_goal'] = latent_goal
    return obs

if __name__ == "__main__":
    use_imagenet = "imagenet" in sys.argv
    variant = dict(
        env_class=SawyerReachXYZEnv,
        env_kwargs=dict(
            action_mode="position", 
            max_speed = 0.05, 
            camera="sawyer_head"
        ),
        # algo_kwargs=dict(
        #     num_epochs=3000,
        #     max_path_length=20,
        #     batch_size=128,
        #     num_eval_steps_per_epoch=1000,
        #     num_expl_steps_per_train_loop=1000,
        #     num_trains_per_train_loop=1000,
        #     min_num_steps_before_training=1000,
        # ),
        algo_kwargs=dict(
            num_epochs=3000,
            max_path_length=10,
            batch_size=5,
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=10,
            num_trains_per_train_loop=10,
            min_num_steps_before_training=10,
        ),
        model_kwargs=dict(
            decoder_distribution='gaussian_identity_variance',
            input_channels=3,
            imsize=224,
            architecture=dict(
                hidden_sizes=[200, 200],
            ),
            delta_features=True,
            pretrained_features=False,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            demo_path="/home/anair/ros_ws/src/railrl-private/demo_v2_2.npy",
            add_demo_latents=True,
            bc_num_pretrain_steps=100,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),

        save_video=True,
        dump_video_kwargs=dict(
            save_period=1,
            # imsize=(3, 500, 300),
        )
    )

    ptu.set_gpu_mode("gpu")

    representation_size = 128
    output_classes = 20

    model_class = variant.get('model_class', TimestepPredictionModel)
    model = model_class(
        representation_size,
        # decoder_output_activation=decoder_activation,
        output_classes=output_classes,
        **variant['model_kwargs'],
    )
    # model = torch.nn.DataParallel(model)

    imagenets = [True, False]
    reg_types = ["regression_distance", "latent_distance"]
    for use_imagenet in [True]:
        for reg_type in ["latent_distance"]:
            print("Processing with imagenet: %s, type: %s" %(str(use_imagenet), reg_type))
            if use_imagenet:
                model_path = "/home/anair/data/s3doodad/facebook/models/rfeatures/multitask1/run2/id0/itr_0.pt" # imagenet
            else:
                model_path = "/home/anair/data/s3doodad/facebook/models/rfeatures/multitask1/run2/id2/itr_4000.pt"

            # model = load_local_or_remote_file(model_path)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            model.to(ptu.device)
            model.eval()

            for color in ["grey"]:
                reference_path = "demos/door_demos_v3/demo_v3_%s_0.pkl"%color
                traj = np.load("demos/door_demos_v3/demo_v3_%s_0.pkl"%color, allow_pickle=True)[0]

                goal_image_flat = traj["observations"][-1]["image_observation"]
                goal_image = goal_image_flat.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
                # goal_image = goal_image[:, ::-1, :, :].copy() # flip bgr
                goal_image = goal_image[:, :, 60:300, 30:470]
                goal_image_pt = ptu.from_numpy(goal_image)
                save_image(goal_image_pt.data.cpu(), 'goal.png', nrow=1)
                goal_latent = model.encode(goal_image_pt).detach().cpu().numpy().flatten()

                initial_image_flat = traj["observations"][0]["image_observation"]
                initial_image = initial_image_flat.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
                # initial_image = initial_image[:, ::-1, :, :].copy() # flip bgr
                initial_image = initial_image[:, :, 60:300, 30:470]
                initial_image_pt = ptu.from_numpy(initial_image)
                save_image(initial_image_pt.data.cpu(), 'initial.png', nrow=1)
                initial_latent = model.encode(initial_image_pt).detach().cpu().numpy().flatten()
                print("Finished initial_latent")
                reward_params = dict(
                    goal_latent=goal_latent,
                    initial_latent=initial_latent,
                    goal_image=goal_image_flat,
                    initial_image=initial_image_flat,
                    # type="latent_distance"
                    # type="regression_distance"
                    type=reg_type
                )
                config_params = dict(
                    initial_type="use_initial_from_trajectory",
                    # initial_type="use_initial_from_trajectory",
                    # goal_type="use_goal_from_trajectory",
                    goal_type="",
                    use_initial=True
                )

                env = variant['env_class'](**variant['env_kwargs'])
                env = ImageEnv(env,
                    recompute_reward=False,
                    transpose=True,
                    image_length=450000,
                    reward_type="image_distance",
                    # init_camera=sawyer_pusher_camera_upright_v2,
                )
                env = EncoderWrappedEnv(env, model, reward_params, config_params)
                print("Finished creating env")
                demo_paths=["/home/anair/ros_ws/src/railrl-private/demos/rlbench/demo_left_%i.mp4" % i for i in range(10)]
                # demo_paths+=["/home/anair/ros_ws/src/railrl-private/demos/rlbench/demo_left_%i.pkl" % i for i in range(10)]

            # processed_demo_path = "/home/anair/ros_ws/src/railrl-private/demos/door_demos_v3/processed_demos_imagenet2.pkl" # use this for imagenet
            # processed_demo_path = "/home/anair/ros_ws/src/railrl-private/demos/door_demos_v3/processed_demos_imagenet_jitter2.pkl"
                if use_imagenet:
                    processed_demo_path = "/home/anair/ros_ws/src/railrl-private/demos/rlbench/demo_right_%s_imagenet_jitter2.pkl" % color
                else:
                    processed_demo_path = "/home/anair/ros_ws/src/railrl-private/demos/door_demos_v3/processed_demos_%s_jitter2.pkl" % color
                name = color
                if use_imagenet:
                    name = "_imagenet_%s"%color
                name = "demos/rlbench/%s_%s"%(name,reward_params["type"])
                print("Loading demos for: ", name)
                load_demos(demo_paths, processed_demo_path, reference_path, name)
                demo_trajectory_rewards = []