"""See https://github.com/aravindr93/hand_dapg for setup instructions"""

import os.path as osp
import pickle
from railrl.core import logger
from railrl.misc.asset_loader import load_local_or_remote_file
from railrl.envs.wrappers import RewardWrapperEnv
import railrl.torch.pytorch_util as ptu

import gym
import numpy as np
import os
import sys
import tensorflow as tf

import learning.awr_agent as awr_agent

import mj_envs

AWR_CONFIGS = {
    "Ant-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.2,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
        "offpolicy_data_kwargs": dict(
            demo_path="/home/ashvin/data/s3doodad/demos/icml2020/mujoco/ant.npy",
        )
    },

    "HalfCheetah-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
        "offpolicy_data_kwargs": dict(
            demo_path="/home/ashvin/data/s3doodad/demos/icml2020/mujoco/half-cheetah.npy",
        )
    },

    "Hopper-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.0001,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "Humanoid-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00001,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "LunarLander-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.0005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_l2_weight": 0.001,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 100000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "Walker2d-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.000025,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "rlbench":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 10.0,
    },

    "pen-v0":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 100.0,

        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="/home/ashvin/data/s3doodad/demos/icml2020/hand/pen2_sparse.npy",
                obs_dict=True,
                is_demo=True,
            ),
            dict(
                path="/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc_sparse4.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },

    "pen-v0":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 100.0,

        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="/home/ashvin/data/s3doodad/demos/icml2020/hand/pen2_sparse.npy",
                obs_dict=True,
                is_demo=True,
            ),
            dict(
                path="/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc_sparse4.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },
}

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

def build_env(env_id):
    assert(env_id is not ""), "Unspecified environment."
    env = gym.make(env_id)
    return env

def build_agent(env, env_id):
    agent_configs = {}
    if (env_id in AWR_CONFIGS):
        agent_configs = AWR_CONFIGS[env_id]

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    agent = awr_agent.AWRAgent(env=env, sess=sess, **agent_configs)

    return agent

default_variant = dict(
    train = True,
    test = True,
    max_iter = 100,
    test_episodes = 32,
    output_iters = 50,
    visualize = False,
    model_file = "",
)

def experiment(user_variant):
    variant = default_variant.copy()
    variant.update(user_variant)

    if ptu.gpu_enabled():
        enable_gpus("0")

    env_id = variant["env"]
    env = build_env(env_id)

    agent = build_agent(env, env_id)
    agent.visualize = variant["visualize"]
    model_file = variant.get("model_file")
    if (model_file is not ""):
        agent.load_model(model_file)

    log_dir = logger.get_snapshot_dir()
    if (variant["train"]):
        agent.train(max_iter=variant["max_iter"],
                    test_episodes=variant["test_episodes"],
                    output_dir=log_dir,
                    output_iters=variant["output_iters"])
    else:
        agent.eval(num_episodes=variant["test_episodes"])

    return

if __name__ == "__main__":
    main(sys.argv)














