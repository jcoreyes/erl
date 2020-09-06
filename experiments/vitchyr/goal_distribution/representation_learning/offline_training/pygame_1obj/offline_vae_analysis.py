from pathlib import Path
import json

import gym
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from multiworld import register_all_envs
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv, DictEncoderWrappedEnv
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.torch.sets.debug import (
    compute_reward_correlations,
    sample_states,
)
from rlkit.torch.sets import rewards
from rlkit.torch.sets.set_creation import create_sets
import pandas as pd

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
            # dict(
            #     version='project_onto_axis',
            #     axis_idx_to_value={
            #         0: 3,
            #         1: 3,
            #     },
            # ),
            # dict(
            #     version='project_onto_axis',
            #     axis_idx_to_value={
            #         0: -2,
            #     },
            # ),
            # dict(
            #     version='project_onto_axis',
            #     axis_idx_to_value={
            #         2: -3,
            #         3: 3,
            #     },
            # ),
            # dict(
            #     version='project_onto_axis',
            #     axis_idx_to_value={
            #         3: -4,
            #     },
            # ),
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


def print_settings(variant_path):
    print('')
    print('--------- settings ----------')
    print('\tpath', variant_path)
    params = json.load(open(variant_path, 'r'))
    for k in [
        'use_fancy_architecture',
        'set_loss_weight',
        'y_values',
        'decoder_distribution',
        'unique_id',
    ]:
        print('\t', k, find_item(params, k))


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
    print_settings(variant_path)
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

    img_env = InsertImageEnv(state_env, renderer=renderer)
    env = DictEncoderWrappedEnv(
        img_env,
        vae,
        encoder_input_key='image_observation',
        encoder_output_remapping={'posterior_mean': 'latent_observation'},
    )
    analyze(sets, vae, env, **kwargs)


def analyze_from_rl(snapshot_path, **kwargs):
    data = torch.load(open(snapshot_path, "rb"))
    env = data['evaluation/env']
    vae = env.model
    sets = env.context_distribution.sets
    env = DictEncoderWrappedEnv(
        env.env.wrapped_env,
        vae,
        encoder_input_key='image_observation',
        encoder_output_remapping={'posterior_mean': 'latent_observation'},
    )
    analyze(sets, vae, env, **kwargs)


def analyze(
        sets,
        vae,
        env,
        n_obs=1024,
        save_dir=None,
):
    vae.to(ptu.device)
    states = sample_states(env, n_obs)
    state_obs = states['state_observation']
    latent_obs = states['latent_observation']

    y = np.array(state_obs)
    x = np.array(latent_obs)
    A = np.linalg.inv(x.T @ x) @ x.T @ y
    y_hat = x @ A

    squared_errors = (y - y_hat) ** 2
    mse_results = []
    for i in range(squared_errors.shape[1]):
        mse_i = np.mean(squared_errors[:, i])
        variance_i = y[:, i].std() ** 2
        # print('mse dim', i, ':', mse_i)
        mse_results.append([i, mse_i / variance_i, mse_i])
        # print('normalized mse dim', i, ':', mse_i / variance_i)
    print('normalized MSEs:', mse_results)

    reward_fns = dict(
        mahalanobis_distance=rewards.NormalLikelihoodRewardFn(
            observation_key='latent_observation',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            drop_log_det_term=True,
            sqrt_reward=True,
        ),
        proper_likelihood=rewards.NormalLikelihoodRewardFn(
            observation_key='latent_observation',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            drop_log_det_term=False,
            use_proper_scale_diag=True,
            sqrt_reward=False,
        ),
        cross_entropy_prior_to_obs=rewards.CrossEntropyRewardFn(
            observation_mean_key='posterior_mean',
            observation_covariance_key='posterior_covariance',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            obs_to_prior_direction=False,
        ),
        kl_prior_to_obs=rewards.CrossEntropyRewardFn(
            observation_mean_key='posterior_mean',
            observation_covariance_key='posterior_covariance',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            use_kl_not_ce=True,
            obs_to_prior_direction=False,
        ),
        cross_entropy_obs_to_prior=rewards.CrossEntropyRewardFn(
            observation_mean_key='posterior_mean',
            observation_covariance_key='posterior_covariance',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            obs_to_prior_direction=True,
        ),
        kl_obs_to_prior=rewards.CrossEntropyRewardFn(
            observation_mean_key='posterior_mean',
            observation_covariance_key='posterior_covariance',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            use_kl_not_ce=True,
            obs_to_prior_direction=True,
        ),
    )

    correlation_stats = {}
    num_sets = 0
    for name, reward_fn in reward_fns.items():
        correlations = compute_reward_correlations(reward_fn, sets, states, vae)
        print(name, 'correlations', correlations)
        correlation_stats[name] = correlations
        num_sets = len(correlations)

    if save_dir is not None:
        correlation_stats['dimension'] = list(range(num_sets))
        corr_results = pd.DataFrame(correlation_stats)
        corr_save_path = str(save_dir / 'reward_correlation.csv')
        corr_results.to_csv(corr_save_path, index=False, header=True, sep=",")
        print('saving to', corr_save_path)

        mse_results = pd.DataFrame(
            mse_results,
            columns=['dimension', 'normalized_mse', 'mse'],
        )
        mse_save_path = str(save_dir / 'linear_state_prediction_mse.csv')
        mse_results.to_csv(mse_save_path, index=False, header=True, sep=",")
        print('saving to', mse_save_path)


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    # base_dir = '/home/vitchyr/mnt/log/20-08-21-exp7-five-mutually-exclusive-sets-with-decent-vae-settings-take2/'
    base_dir = '/home/vitchyr/mnt/log/20-08-28-exp4-global-fixed-variance/'

    for path in Path(base_dir).rglob('params.pt'):
        print(path)
        snapshot_path = path.absolute()
        analyze_from_vae(
            str(snapshot_path),
            save_dir=snapshot_path.parent,
        )

    # snapshot_path = '/home/vitchyr/mnt/log/20-08-13-create-tmp-vae/20-08-13-create-tmp-vae_2020_08_13_16_09_19_id000--s172096/params.pkl'
    # analyze_from_rl(snapshot_path)
