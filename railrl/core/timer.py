import time


class Timer:
    def __init__(self, return_global_times=False):
        self.reset()
        self.global_start_time = self.epoch_start_time
        self.return_global_times = return_global_times

    def reset(self):
        self.stamps = {}
        self.epoch_start_time = time.time()
        self.last_stamp_time = self.epoch_start_time

    def stamp(self, name, unique=True):
        if unique:
            assert name not in self.stamps.keys()
        cur_time = time.time()
        if name not in self.stamps:
            self.stamps[name] = 0.0
        self.stamps[name] += (cur_time - self.last_stamp_time)
        self.last_stamp_time = cur_time

    def get_times(self):
        global_times = {}
        if self.return_global_times:
            cur_time = time.time()
            global_times['global_time'] = (cur_time - self.global_start_time)
            global_times['epoch_time'] = (cur_time - self.epoch_start_time)
        return {
            **self.stamps.copy(),
            **global_times,
        }


timer = Timer()
