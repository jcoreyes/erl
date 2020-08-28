"""

railrl git hash: a36662443 on `probabilistc-goal-reaching` branch.
multiworld git hash off of 200604-visuals-for-neurips branch: ee584c83b5c5ebd425075d3633bbf1f3f22a7091
"""
import numpy as np
import pickle
from collections import OrderedDict
from functools import partial
import moviepy.editor as mpy

from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from railrl.visualization.video import dump_video

from railrl.envs.contextual import (
    ContextualEnv, ContextualRewardFn,
    delete_info,
)
import railrl.samplers.rollout_functions as rf
from railrl.envs.contextual.goal_conditioned import (
    AddImageDistribution,
    PresampledDistribution,
)
from railrl.envs.images import EnvRenderer
from railrl.launchers.contextual.util import get_save_video_function
from railrl.launchers.experiments.vitchyr.probabilistic_goal_reaching.visualize import \
    InsertDebugImagesEnv


def main(
        checkpoint_file,
        save_file,
        n=5,
        presampled_goal_distribution=None,
):

    data = pickle.load(open(checkpoint_file, "rb"))
    env = data['evaluation/eval/env']
    policy = data['evaluation/eval/policy']
    observation_key = data['evaluation/eval/observation_key']
    context_keys_for_policy = data['evaluation/eval/context_keys_for_policy']
    desired_goal_key = context_keys_for_policy[0]
    horizon = 100

    def set_goal_for_visualization(env, policy, o):
        goal = o[desired_goal_key]
        env.unwrapped.goal = goal

    rollout_function = partial(
        rf.contextual_rollout,
        max_path_length=horizon,
        observation_key=observation_key,
        context_keys_for_policy=context_keys_for_policy,
        reset_callback=set_goal_for_visualization,
    )
    video_renderer = EnvRenderer(
        init_camera=sawyer_init_camera_zoomed_in,
        height=256,
        output_image_format="HWC",
        width=256,
        normalize_image=False,
    )
    renderers = OrderedDict(
        image_observation=video_renderer,
    )


    state_env = env.env

    if presampled_goal_distribution:
        base_dist = presampled_goal_distribution
        base_dist._i = np.array([0])
    else:
        base_dist = env.context_distribution
        base_dist._i = np.array([0])

    goal_distribution = AddImageDistribution(
        env=state_env,
        base_distribution=base_dist,
        image_goal_key='image_desired_goal',
        renderer=video_renderer,
    )
    context_env = ContextualEnv(
        state_env,
        context_distribution=goal_distribution,
        reward_fn=env._reward_fn,
        observation_key=observation_key,
        update_env_info_fn=delete_info,
    )
    env = InsertDebugImagesEnv(
        context_env,
        renderers=renderers,
    )

    def get_goals_and_obs():
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )
        goals = []
        obs = []
        for _  in range(n):
            for i_in_path, d in enumerate(path['full_observations']):
                goals.append(d['image_desired_goal'])
                obs.append(d['image_observation'])
        return goals, obs

    def create_video(imgs):
        return mpy.ImageSequenceClip(imgs, fps=20)

    def create_goal_and_img_videos():
        goals, obs = get_goals_and_obs()
        goal_video = create_video(goals)

        goal_txt = mpy.TextClip(
            "goal", fontsize=70, color='white', font='Roboto-Light',
        ).set_position(
            ('center', 'bottom'),
        ).set_duration(
            goal_video.duration
        )
        captioned_goal_video = mpy.CompositeVideoClip([goal_video, goal_txt])

        rollout_video = create_video(obs)
        rollout_txt = mpy.TextClip(
            "state", fontsize=70, color='white', font='Roboto-Light',
        ).set_position(
            ('center', 'bottom'),
        ).set_duration(
            goal_video.duration
        )
        captioned_rollout = mpy.CompositeVideoClip([rollout_video, rollout_txt])
        return [captioned_goal_video, captioned_rollout]

    all_videos = [
        create_goal_and_img_videos() for _ in range(4)
    ]
    # import ipdb; ipdb.set_trace()
    combined_clip = mpy.clips_array(
        all_videos,
        cols_widths=[258]*2,
        rows_widths=[258]*4,
    )
    # combined_clip.write_videofile('/home/vitchyr/tmp.mp4')
    combined_clip.write_gif(save_file)
    combined_clip.write_videofile(save_file.replace('gif', 'mp4'))
    return presampled_goal_distribution


if __name__ == '__main__':
    pre = main(
        '/home/vitchyr/mnt/log/20-06-01-pgr--sawyer--exp-3--push-laplace-redo-learned-laplace-take2/20-06-01-pgr--sawyer--exp-3--push-laplace-redo-learned-laplace-take2_2020_06_01_15_24_28_id000--s528845/params.pkl',
        '/home/vitchyr/mnt/log/manually-generated/probabilistic-goal-reaching/sawyer_push/gcac_5_rollouts.gif',
        n=5,
    )
    main(
        '/home/vitchyr/mnt/log/20-06-01-pgr--sawyer--exp-3--push-laplace-redo-learned-laplace-take2/20-06-01-pgr--sawyer--exp-3--push-laplace-redo-learned-laplace-take2_2020_06_01_15_24_28_id002--s141740/params.pkl',
        '/home/vitchyr/mnt/log/manually-generated/probabilistic-goal-reaching/sawyer_push/prob_reward_5_rollouts.gif',
        n=5,
        presampled_goal_distribution=pre,
    )
