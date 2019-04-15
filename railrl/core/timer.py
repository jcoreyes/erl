import time


class Timer:
    def reset(self):
        self.stamps = {}
        self.start_time = time.time()

    def stamp(self, name, unique=True):
        if unique:
            assert name not in self.stamps.keys()
        self.stamps[name] = (time.time() - self.start_time)

    def get_times(self):
        return self.stamps.copy()


timer = Timer()
