from rlkit.envs.memory.hidden_cartpole import HiddenCartpoleEnv
import numpy as np
import time

from rlkit.exploration_strategies.ou_strategy import OUStrategy

env = HiddenCartpoleEnv()

all_returns = []
es = OUStrategy(env, max_sigma=0.01, min_sigma=0.01)
while True:
    obs = env.reset()
    # print("init obs", obs)
    zero_action = np.zeros(1)
    action = zero_action
    last_reward_t = 0
    print("---------- RESET ----------")
    returns = 0
    for t in range(50):
        # action = es.get_action_from_raw_action(zero_action)
        action = np.array([1])
        obs, reward, done, info = env.step(action)
        # print("action", action)
        # print("obs", obs)
        print("reward", reward)
        print("done", done)
        # print("target", target)
        returns += reward
        time.sleep(0.1)
        env.render()
    print("Returns", returns)
    all_returns.append(returns)
    print("Returns Mean", np.mean(all_returns))
    print("Returns Std", np.std(all_returns, axis=0))
