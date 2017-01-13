# from envs.env_utils import register_environments
from gym.envs.registration import register

register(id='OneDPoint-v0', entry_point='envs.oned_point:OneDPoint')
register(id='TwoDPoint-v0', entry_point='envs.twod_point:TwoDPoint')
register(id='TwoDPointRandomInit-v0',
         entry_point='envs.twod_point_random_init:TwoDPointRandomInit')
register(id='TwoDMaze-v0', entry_point='envs.twod_maze:TwoDMaze')
