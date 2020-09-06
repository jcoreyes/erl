from multiworld import register_all_envs
from rlkit.torch.sets.debug import save_reward_visualizations

register_all_envs()
from pathlib import Path

import gym
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from multiworld import register_all_envs
from rlkit.envs.images import EnvRenderer
from rlkit.torch.sets import rewards
from rlkit.torch.sets.set_creation import create_sets

register_all_envs()


def find_item(obj, key):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            result = find_item(v, key)
            if result is not None:
                return result

def make_custom_sets(env, renderer):
    sets = create_sets(
        env,
        renderer,
        num_samples_per_set=128,
        set_configs=[
            dict(
                version='project_onto_axis',
                axis_idx_to_value={
                    0: 3,
                    1: 3,
                },
            ),
            dict(
                version='project_onto_axis',
                axis_idx_to_value={
                    0: -2,
                },
            ),
            dict(
                version='project_onto_axis',
                axis_idx_to_value={
                    2: -3,
                    3: 3,
                },
            ),
            dict(
                version='project_onto_axis',
                axis_idx_to_value={
                    3: -4,
                },
            ),
            dict(
                version='move_a_to_b',
                offsets_from_b=(4, 0),
                a_axis_to_b_axis={
                    0: 2,
                    1: 3,
                },
            ),
        ],
    )
    return sets



def analyze_from_vae(
        snapshot_path,
        latent_observation_key='latent_observation',
        mean_key='latent_mean',
        covariance_key='latent_covariance',
        image_observation_key='image_observation',
        **kwargs
):
    data = torch.load(open(snapshot_path, "rb"))
    variant_path = snapshot_path.replace('params.pt', 'variant.json')
    vae = data['trainer/vae']
    state_env = gym.make('OneObject-PickAndPlace-BigBall-RandomInit-2D-v1')
    renderer = EnvRenderer()
    sets = make_custom_sets(state_env, renderer)
    reward_fn, _ = rewards.create_normal_likelihood_reward_fns(
        latent_observation_key=latent_observation_key,
        mean_key=mean_key,
        covariance_key=covariance_key,
        reward_fn_kwargs=dict(
            drop_log_det_term=True,
            sqrt_reward=True,
        ),
    )
    save_reward_visualizations(sets, vae, state_env, renderer, **kwargs)

def analyze_from_rl(snapshot_path, **kwargs):
    data = torch.load(open(snapshot_path, "rb"))
    env = data['evaluation/env']
    vae = env.model
    renderer = env.env.wrapped_env.renderers['image_observation']
    sets = env.context_distribution.sets
    state_env = env.env.wrapped_env.env
    save_reward_visualizations(sets, vae, state_env, renderer, **kwargs)


def batch_wise_eval(fn, batch, max_batch_size):
    num_elems = batch.shape[0]
    if num_elems <= max_batch_size:
        return fn(batch)

    num_b = num_elems // max_batch_size
    outputs = []
    for i in range(num_b):
        start_i = i * max_batch_size
        end_i = min((i + 1) * max_batch_size, num_elems)
        mini_batch = batch[start_i:end_i, ...]
        outputs.append(fn(mini_batch))
    return np.concatenate(outputs, axis=0)


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    base_dir = '/home/vitchyr/mnt/log/20-08-21-exp7-five-mutually-exclusive-sets-with-decent-vae-settings-take2/'

    for path in Path(base_dir).rglob('params.pkl'):
        snapshot_path = path.absolute()
        analyze_from_rl(
            str(snapshot_path),
            save_dir=snapshot_path.parent,
        )
    # base_dir = '/home/vitchyr/mnt/log/20-08-28-exp4-global-fixed-variance/'
    #
    # for path in Path(base_dir).rglob('params.pt'):
    #     print(path)
    #     snapshot_path = path.absolute()
    #     analyze_from_vae(
    #         str(snapshot_path),
    #         save_dir=snapshot_path.parent,
    #     )


    # snapshot_path = '/home/vitchyr/mnt/log/20-08-13-create-tmp-vae/20-08-13-create-tmp-vae_2020_08_13_16_09_19_id000--s172096/params.pkl'
    # analyze_from_rl(snapshot_path)
