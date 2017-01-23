from rllab.envs.gym_env import GymEnv


def gym_env(name):
    return GymEnv(name,
                  record_video=False,
                  log_dir='/tmp/gym-test',  # Ignore gym log.
                  record_log=False)