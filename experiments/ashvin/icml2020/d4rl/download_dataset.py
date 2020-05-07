env_names = [ # D4RL envs
"maze2d-open-v0", "maze2d-umaze-v0", "maze2d-medium-v0", "maze2d-large-v0",
"maze2d-open-dense-v0", "maze2d-umaze-dense-v0", "maze2d-medium-dense-v0", "maze2d-large-dense-v0",
"antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-diverse-v0",
"antmaze-medium-play-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
# "pen-demos-v0", "pen-cloned-v0", "pen-expert-v0", "hammer-demos-v0", "hammer-cloned-v0", "hammer-expert-v0",
# "door-demos-v0", "door-cloned-v0", "door-expert-v0", "relocate-demos-v0", "relocate-cloned-v0", "relocate-expert-v0",
"halfcheetah-random-v0", "halfcheetah-medium-v0", "halfcheetah-expert-v0", "halfcheetah-mixed-v0", "halfcheetah-medium-expert-v0",
"walker2d-random-v0", "walker2d-medium-v0", "walker2d-expert-v0", "walker2d-mixed-v0", "walker2d-medium-expert-v0",
"hopper-random-v0", "hopper-medium-v0", "hopper-expert-v0", "hopper-mixed-v0", "hopper-medium-expert-v0"
]

import gym
import d4rl

for env_name in env_names:
      env = gym.make(env_name)
      env.get_dataset()
