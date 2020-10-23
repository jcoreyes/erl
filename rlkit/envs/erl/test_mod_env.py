import os
import xml.etree.ElementTree as ET
#import ipdb
import numpy as np
from rlkit.envs.erl import (
    HalfCheetahEnv,
    AntEnv,
    Walker2dEnv,
    InvertedDoublePendulumEnv,
    HopperEnv,
    HumanoidEnv,
    SwimmerEnv,
)

if __name__ == '__main__':
    mods = dict(gravity=1.2,
                friction=1.2,
                ctrlrange=1.2,
                gear=1.2
                )
    # friction is only standard in geom for ant, half_cheetah
    # not standord for hooper, could apply to foot or all geoms
    #print(create_new_xml('ant.xml', mods))
    for env_cls in (
        HalfCheetahEnv,
        AntEnv,
        Walker2dEnv,
        InvertedDoublePendulumEnv,
        HopperEnv,
        HumanoidEnv,
        SwimmerEnv,
    ):
        #env = env_cls(mods)
        env = HumanoidEnv(mods)
        env.reset()
        for i in range(10000):
            env.step(env.action_space.sample())
            env.render()
