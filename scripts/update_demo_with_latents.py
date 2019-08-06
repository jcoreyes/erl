# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv

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

demo_trajectory_rewards = []

def load_path(path):
    final_achieved_goal = path["observations"][-1]["state_achieved_goal"].copy()

    print("loading path, length", len(path["observations"]), len(path["actions"]))
    H = min(len(path["observations"]), len(path["actions"]))
    rewards = []

    for i in range(H):
        ob = path["observations"][i]
        action = path["actions"][i]
        reward = path["rewards"][i]
        next_ob = path["next_observations"][i]
        terminal = path["terminals"][i]
        agent_info = path["agent_infos"][i]
        env_info = path["env_infos"][i]

        # goal = path["goal"]["state_desired_goal"][0, :]
        # import ipdb; ipdb.set_trace()
        # print(goal.shape, ob["state_observation"])
        # state_observation = np.concatenate((ob["state_observation"], goal))
        # action = action[:2]

        # update_obs_with_latent(ob)
        # update_obs_with_latent(next_ob)
        env._update_obs(ob)
        env._update_obs(next_ob)
        reward = env.compute_reward(
            action,
            next_ob,
        )
        path["rewards"][i] = reward
        # reward = np.array([reward])
        # terminal = np.array([terminal])

        print(reward)
        rewards.append(reward)
    demo_trajectory_rewards.append(rewards)

def load_demos(demo_path, ):
    data = load_local_or_remote_file(demo_path)
    for path in data:
        load_path(path)
    processed_demo_path = demo_path[:-4] + "_processed.npy"
    np.save(processed_demo_path, data)

    plt.figure(figsize=(8, 8))
    for r in demo_trajectory_rewards:
        plt.plot(r)
    plt.savefig("demo_rewards.png")

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
            demo_path="/home/lerrel/ros_ws/src/railrl-private/demo_v2_2.npy",
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

    model_path = "/home/lerrel/data/s3doodad/facebook/models/rfeatures/multitask1/run2/id2/itr_4000.pt"
    # model = load_local_or_remote_file(model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(ptu.device)
    model.eval()

    traj = np.load("demo_v4.npy", allow_pickle=True)[0]

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

    reward_params = dict(
        goal_latent=goal_latent,
        initial_latent=initial_latent,
        goal_image=goal_image_flat,
        initial_image=initial_image_flat,
    )

    env = variant['env_class'](**variant['env_kwargs'])
    env = ImageEnv(env,
        recompute_reward=False,
        transpose=True,
        image_length=450000,
        reward_type="image_distance",
        # init_camera=sawyer_pusher_camera_upright_v2,
    )
    env = EncoderWrappedEnv(env, model, reward_params)

    demo_path="/home/lerrel/ros_ws/src/railrl-private/demo_v4.npy"
    load_demos(demo_path)