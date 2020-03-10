# first to start the nameserver start: python -m Pyro4.naming

import Pyro4
from threading import Thread
import time
import numpy as np
from railrl.launchers import config
# HOSTNAME = "192.168.0.102"

Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle','json', 'marshal', 'serpent'])
Pyro4.config.SERIALIZER='pickle'

device_state = None

@Pyro4.expose
class DeviceState(object):
    state = None

    def get_state(self):
        return device_state
        # return self.state

    def set_state(self, state):
        # print("set", state)
        # self.state = state
        global device_state
        device_state = state

class SpaceMouseExpert:
    def __init__(self, xyz_dims=3, xyz_remap=[0, 1, 2], xyz_scale=[1, 1, 1]):
        """TODO: fill in other params"""
        self.xyz_dims = xyz_dims
        self.xyz_remap = np.array(xyz_remap)
        self.xyz_scale = np.array(xyz_scale)
        self.thread = Thread(target = start_server)
        self.thread.daemon = True
        self.thread.start()
        self.device_state = DeviceState()

    def get_action(self, obs):
        """Must return (action, valid, reset, accept)"""
        state = self.device_state.get_state()
        print(state)
        if state is None:
            return None, False, False, False

        dpos, rotation, accept, reset = (
            state["dpos"],
            state["rotation"],
            state["left_click"],
            state["right_click"],
        )

        xyz = dpos[self.xyz_remap] * self.xyz_scale

        a = xyz[:self.xyz_dims]

        valid = not np.all(np.isclose(a, 0))

        return (a, valid, reset, accept)


def start_server():
    daemon = Pyro4.Daemon(config.SPACEMOUSE_HOSTNAME)                # make a Pyro daemon
    ns = Pyro4.locateNS()                  # find the name server
    uri = daemon.register(DeviceState)   # register the greeting maker as a Pyro object
    ns.register("example.greeting", uri)   # register the object with a name in the name server

    print("Server ready.")
    daemon.requestLoop()                   # start the event loop of the server to wait for calls

if __name__ == "__main__":
    expert = SpaceMouseExpert()

    for i in range(100):
        time.sleep(1)
        print(expert.get_action(None))
