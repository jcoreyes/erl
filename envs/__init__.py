from gym.envs.registration import register

register(id='AxeOneDPoint-v0', entry_point='envs.oned_point:OneDPoint')
register(id='AxeTwoDPoint-v0', entry_point='envs.twod_point:TwoDPoint')
register(id='AxeTwoDMaze-v0', entry_point='envs.twod_maze:TwoDMaze')
