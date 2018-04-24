from railrl.envs.mujoco.sawyer_gripper_env import SawyerPushEnv, SawyerPushXYEnv

from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.policies.simple import ZeroPolicy
import numpy as np

print("making env")
env = SawyerPushXYEnv(randomize_goals=True, frame_skip=50)

policy = ZeroPolicy(env.action_space.low.size)
es = OUStrategy(
    env.action_space,
    theta=1
)
policy = exploration_policy = PolicyWrappedWithExplorationStrategy(
    exploration_strategy=es,
    policy=policy,
)
print("starting rollout")
while True:
    obs = env.reset()
    last_reward_t = 0
    returns = 0
    for t in range(300):
        # action = env.action_space.sample()*10
        action, _ = policy.get_action(None)
        delta = (env.get_block_pos() - env.get_endeff_pos())[:2]
        action[:2] = delta * 100
        # action[1] -= 0.05
        action = np.sign(action)
        action += np.random.normal(size=action.shape) * 0.2
        error = np.linalg.norm(delta)
        print("action", action)
        print("error", error)
        if error < 0.04:
            action[1] += 10
        obs, reward, done, info = env.step(action)

        env.render()
        # print("action", action)
        if done:
            break
    print("new episode")
