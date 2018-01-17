import os

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)


def gym_env(name):
    from rllab.envs.gym_env import GymEnv
    return GymEnv(name,
                  record_video=False,
                  log_dir='/tmp/gym-test',  # Ignore gym log.
                  record_log=False)