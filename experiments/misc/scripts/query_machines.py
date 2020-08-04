from railrl.misc.exp_util import query_machines
import numpy as np

if __name__ == '__main__':
    free_machines_info = query_machines()

    machines_names = list(free_machines_info.keys())
    machines_names.sort()

    print("Number of free GPUs:",
          np.sum([len(free_machines_info[k]["gpu_ids"]) for k in free_machines_info.keys()]))
    for machine in machines_names:
        print(machine, free_machines_info[machine]["gpu_ids"], free_machines_info[machine]["mem"], "Gb")