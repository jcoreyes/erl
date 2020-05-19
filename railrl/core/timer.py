import time


class Timer:
    def __init__(self, return_global_times=False):
        self.stamps = None
        self.epoch_start_time = None
        self.last_stamp_time = None
        self.start_times = None
        self.global_start_time = time.time()
        self._return_global_times = return_global_times

        self.reset()

    def reset(self):
        self.stamps = {}
        self.start_times = {}
        self.epoch_start_time = time.time()
        self.last_stamp_time = self.epoch_start_time

    def stamp(
        self,
        name,
        unique=True,
        start_time=None,
        update_last_stamp_time=True,
    ):
        if unique:
            assert name not in self.stamps.keys()

        if start_time is None:
            start_time = self.last_stamp_time

        cur_time = time.time()
        if name not in self.stamps:
            self.stamps[name] = 0.0
        self.stamps[name] += (cur_time - start_time)

        if update_last_stamp_time:
            self.last_stamp_time = cur_time

    def get_times(self):
        global_times = {}
        cur_time = time.time()
        global_times['epoch_time'] = (cur_time - self.epoch_start_time)
        if self._return_global_times:
            global_times['global_time'] = (cur_time - self.global_start_time)
        return {
            **self.stamps.copy(),
            **global_times,
        }

    def stamp_start(self, name):
        self.start_times[name] = time.time()

    def stamp_end(self, name, unique=True, update_last_stamp_time=True):
        assert name in self.start_times
        self.stamp(
            name,
            unique=unique,
            start_time=self.start_times[name],
            update_last_stamp_time=update_last_stamp_time,
        )
        del self.start_times[name]

    @property
    def return_global_times(self):
        return self._return_global_times

    @return_global_times.setter
    def return_global_times(self, value):
        self._return_global_times = value


timer = Timer()
