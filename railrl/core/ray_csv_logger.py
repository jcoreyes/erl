from ray.tune.logger import CSVLogger
import os


class SequentialCSVLogger(CSVLogger):
    """CSVLogger to be used with SequentialRayExperiment

    on receiving a log_dict with next_algo=True, a new csv progress file we be
    used.
    """
    def _init(self):
        self.created_logger = False
        # We need to create an initial self._file to make Ray happy...
        self.setup_new_logger('temp_progress.csv')

    def setup_new_logger(self, csv_fname):
        self.csv_fname = csv_fname
        if self.created_logger:
            self._file.close()
        self.created_logger = True
        progress_file = os.path.join(self.logdir, csv_fname)
        self._continuing = os.path.exists(progress_file)
        self._file = open(progress_file, "a")
        self._csv_out = None

    def on_result(self, result):
        if 'log_fname' in result and result['log_fname'] != self.csv_fname:
            self.setup_new_logger(result['log_fname'])
        super().on_result(result)
