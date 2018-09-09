from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnvYZ
import numpy as np

def presample_goals(n_goals=5000):
    env = SawyerPickAndPlaceEnvYZ(
        hide_goal_markers=True,
        hide_arm=True,
        action_scale=.02,
        reward_type="obj_distance",
        random_init=True,
    )
    env.reset()
    goals = []
    for i in range(n_goals):
        goal = env.sample_goal()
        env.set_to_goal(goal, set_goal=True)
        print(i)
        goals.append(goal['state_desired_goal'])
    goals = np.array(goals)
    np.save('goals.npy', goals)
    return goals