"""Since pickle apparently doesn't know how to pickle methods/classes defined
in __main__
"""
import numpy as np

def PickAndPlace1DEnvObjectOnlyRewardFn(obs_dict, actions, next_obs_dict,
                                        new_contexts, context_key, mask_key):
        achieved_goals = next_obs_dict['state_achieved_goal']
        batch_size = len(achieved_goals)
        desired_goals = new_contexts[context_key]
        mask = new_contexts[mask_key]
        rewards = np.zeros(batch_size)

        for obj_num in range(1, len(mask[0]) + 1):
            achieved_obj = achieved_goals[:, obj_num * 2:(obj_num + 1) * 2]
            desired_obj = desired_goals[:, obj_num * 2:(obj_num + 1) * 2]
            rewards -= mask[:, obj_num - 1] * (
                np.linalg.norm(
                    achieved_obj - desired_obj,
                    2,
                    axis=1
                )
            )
        return rewards


