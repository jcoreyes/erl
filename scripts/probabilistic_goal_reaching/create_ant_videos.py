"""
railrl git hash off of probabilistic-goal-reaching branch: 9c8747c9d
multiworld git hash off of 200604-visuals-for-neurips branch: 38c7e8f
"""
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
from railrl.envs.contextual.goal_conditioned import AddImageDistribution
from railrl.envs.images import EnvRenderer, GymEnvRenderer
from railrl.launchers.contextual.util import get_save_video_function
from railrl.launchers.experiments.vitchyr.probabilistic_goal_reaching.visualize import \
    InsertDebugImagesEnv


def main(
        checkpoint_file,
        save_file,
        unnormalize,
        n=1,
        env=None,
):

    data = pickle.load(open(checkpoint_file, "rb"))
    env = env or data['evaluation/eval/env']
    policy = data['evaluation/eval/policy']
    observation_key = data['evaluation/eval/observation_key']
    context_keys_for_policy = data['evaluation/eval/context_keys_for_policy']
    desired_goal_key = context_keys_for_policy[0]
    import ipdb; ipdb.set_trace()
    horizon = 100

    qpos_weights = env.unwrapped.presampled_qpos.std(axis=0)

    def set_goal_for_visualization(env, policy, o):
        goal = o[desired_goal_key]
        if unnormalize:
            unnormalized_goal = goal * qpos_weights
            env.unwrapped.goal = unnormalized_goal
        else:
            env.unwrapped.goal = goal

    rollout_function = partial(
        rf.contextual_rollout,
        max_path_length=horizon,
        observation_key=observation_key,
        context_keys_for_policy=context_keys_for_policy,
        reset_callback=set_goal_for_visualization,
    )
    video_renderer = GymEnvRenderer(
        init_camera=sawyer_init_camera_zoomed_in,
        height=256,
        output_image_format="HWC",
        width=256,
        normalize_image=False,
    )
    renderers = OrderedDict(
        image_observation=video_renderer,
    )
    def add_images(raw_env):
        state_env = raw_env.env
        # import ipdb; ipdb.set_trace()
        goal_distribution = AddImageDistribution(
            env=state_env,
            base_distribution=raw_env.context_distribution,
            image_goal_key='image_desired_goal',
            renderer=video_renderer,
        )
        context_env = ContextualEnv(
            state_env,
            context_distribution=goal_distribution,
            reward_fn=raw_env._reward_fn,
            observation_key=observation_key,
            update_env_info_fn=delete_info,
        )
        return InsertDebugImagesEnv(
            context_env,
            renderers=renderers,
        )
    # env = add_images(env)
    env = InsertDebugImagesEnv(
        env,
        renderers=renderers,
    )
    def get_obs():
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )
        obs = []
        for i_in_path, d in enumerate(path['full_observations']):
            obs.append(d['image_observation'])
        return obs

    def create_video():
        if n == 1:
            return mpy.ImageSequenceClip(get_obs(), fps=20)
        else:
            return mpy.concatenate_videoclips([
                mpy.ImageSequenceClip(get_obs(), fps=20)
                for _ in range(n)
            ])
        # return rollout_video

    all_videos = [
        [create_video(), create_video()],
        [create_video(), create_video()],
    ]
    # import ipdb; ipdb.set_trace()
    combined_clip = mpy.clips_array(
        all_videos,
        cols_widths=[258]*2,
        rows_widths=[258]*2,
    )
    # combined_clip.write_videofile('/home/vitchyr/tmp.mp4')
    combined_clip.write_gif(save_file)
    combined_clip.write_videofile(save_file.replace('gif', 'mp4'))
    return env.env


if __name__ == '__main__':
    # GCAC
    env = main(
        # '/home/vitchyr/mnt/log/20-06-03-pgr--ant-full-state--exp-12--normalized-ant-dimensions-fixed-htp/20-06-03-pgr--ant-full-state--exp-12--normalized-ant-dimensions-fixed-htp_2020_06_03_16_17_38_id008--s303446/params.pkl',
        '/home/vitchyr/mnt/log/20-06-02-pgr--ant-full-state--exp-12--normalized-ant-dimensions-redo-take2/20-06-02-pgr--ant-full-state--exp-12--normalized-ant-dimensions-redo-take2_2020_06_03_06_22_01_id008--s692730/params.pkl',
        '/home/vitchyr/mnt/log/manually-generated/probabilistic-goal-reaching/ant/gcac_end_1.gif',
        True,
        n=1,
    )
    # sparse
    main(
        # '/home/vitchyr/mnt/log/20-06-02-pgr--ant-full-state--exp-12--normalized-ant-dimensions-redo-take2/20-06-02-pgr--ant-full-state--exp-12--normalized-ant-dimensions-redo-take2_2020_06_03_06_22_00_id014--s278453/params.pkl',
        '/home/vitchyr/mnt/log/20-06-02-pgr--ant-full-state--exp-12--normalized-ant-dimensions-redo-take2/20-06-02-pgr--ant-full-state--exp-12--normalized-ant-dimensions-redo-take2_2020_06_03_06_22_00_id022--s34848/params.pkl',
        '/home/vitchyr/mnt/log/manually-generated/probabilistic-goal-reaching/ant/prob_reward_end_1.gif',
        True,
        n=1,
        env=env,
    )
