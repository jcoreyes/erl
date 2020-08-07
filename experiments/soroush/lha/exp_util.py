import argparse
import os
import numpy as np
import collections
from rlkit.launchers.launcher_util import run_experiment as exp_launcher_function

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--interactive_ssh', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant', action='store_true')
    parser.add_argument('--max_exps_per_instance', type=int, default=None)
    parser.add_argument('--no_video',  action='store_true')
    parser.add_argument('--threshold_for_free_gpu', type=float, default=0.05)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--mem_per_exp', type=float, default=12.0)
    parser.add_argument('--machines_and_gpus', type=str, default=None)
    parser.add_argument('--exp_partition', nargs='*', type=int, default=[])
    return parser.parse_args()

def deep_update(source, overrides):
    '''
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.

    Copied from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def run_experiment(
        exp_function,
        variant,
        args,
        exp_id,
        mount_blacklist=None,
):
    # exp_prefix = get_exp_prefix(args, variant)
    exp_prefix = args.label
    num_exps_for_instances = get_num_exps_for_instances(args)
    for num_exps in num_exps_for_instances:
        run_experiment_kwargs = get_instance_kwargs(args, num_exps, variant)
        exp_launcher_function(
            exp_function,
            variant=variant,
            exp_folder=args.env,
            exp_prefix=exp_prefix,
            exp_id=exp_id,
            snapshot_gap=variant['logger_config']['snapshot_gap'],
            snapshot_mode=variant['logger_config']['snapshot_mode'],
            mount_blacklist=mount_blacklist,

            # base_log_dir=args.env,
            # exp_name=exp_prefix,
            # unpack_variant=False,

            **run_experiment_kwargs
        )

    if args.first_variant:
        exit()

def preprocess_args(args):
    # assert not (args.debug and args.mode == 'ec2')

    if args.max_exps_per_instance is None:
        if args.mode == 'ec2':
            args.max_exps_per_instance = 2
        elif args.mode in ['newton1', 'newton3', 'newton4', 'newton6', 'newton7',
                           'ada', 'alan', 'grace', 'claude', 'lab']:
            args.max_exps_per_instance = 5
        else:
            args.max_exps_per_instance = 1

    if args.mode == 'local' and args.label == '':
        args.label = 'local'
    elif args.mode == 'check_lab':
        args.threshold_for_free_gpu = 0.70
        args.first_variant = True
        args.debug = True
        args.no_video = True

        from time import localtime, strftime
        args.label = 'check-machines-' + strftime('%H-%M', localtime())

        machines = [
            # 'ada',
            # 'alan',
            'newton1',
            # 'newton3',
            # 'newton4',
            # 'newton6',
            # 'newton7',
            # 'grace',
            # 'claude',
        ]  #'newton2', 'newton5', # 'newton4',
        free_machines_info = query_machines(args=args, machines=machines)
        free_machines_and_gpus = []

        for machine in free_machines_info:
            if len(free_machines_info[machine]['gpu_ids']) == 0:
                continue
            most_free_idx = np.argmax(free_machines_info[machine]['gpu_free_mem'])
            gpu_id = free_machines_info[machine]['gpu_ids'][most_free_idx]
            free_machines_and_gpus.append([machine, int(gpu_id)])

        args.num_free_machines_and_gpus = len(free_machines_and_gpus)
        args.free_machines_and_gpus = iter(free_machines_and_gpus)
    elif args.mode == 'lab':
        from itertools import repeat
        import random

        if args.machines_and_gpus is None:
            free_machines_info = query_machines(args=args)
            free_machines_and_gpus_and_counts = []

            free_machines_and_gpu_ids = []
            for machine in free_machines_info:
                for gpu_id in free_machines_info[machine]['gpu_ids']:
                    free_machines_and_gpu_ids.append([machine, gpu_id])

            random.shuffle(free_machines_and_gpu_ids)

            num_exps_for_machine = {m: 0 for m in free_machines_info.keys()}
            for (machine, gpu_id) in free_machines_and_gpu_ids:
                max_exps = int(free_machines_info[machine]['mem'] / args.mem_per_exp)
                num_exps_so_far = num_exps_for_machine[machine]
                num_new_exps = min(max_exps - num_exps_so_far, args.max_exps_per_instance)
                if num_new_exps == 0:
                    continue
                free_machines_and_gpus_and_counts.append([machine, gpu_id, num_new_exps])
                num_exps_for_machine[machine] += num_new_exps

            free_machines_and_gpus = []
            for (machine, gpu_id, count) in free_machines_and_gpus_and_counts:
                for _ in range(count):
                    free_machines_and_gpus.append([machine, int(gpu_id)])
        else:
            import ast
            free_machines_and_gpus = []
            for (machine, gpu_id) in ast.literal_eval(args.machines_and_gpus):
                free_machines_and_gpus.append([machine, int(gpu_id)])

        print('avail exps:', len(free_machines_and_gpus))
        args.free_machines_and_gpus = iter(free_machines_and_gpus)
    elif args.mode in [
        'newton1', 'newton2', 'newton3', 'newton4', 'newton5', 'newton6', 'newton7',
        'ada', 'alan', 'grace', 'claude',
    ]:
        from itertools import repeat
        import random

        free_machines_info = query_machines(machines=[args.mode], args=args)
        free_machines_and_gpus = []

        for machine in free_machines_info:
            max_exps = int(free_machines_info[machine]['mem'] / args.mem_per_exp)
            print(free_machines_info[machine]['gpu_ids'])
            avail_gpu_instances = np.repeat(free_machines_info[machine]['gpu_ids'], args.max_exps_per_instance)[:max_exps]
            for gpu_id in avail_gpu_instances:
                free_machines_and_gpus.append([machine, int(gpu_id)])

        random.shuffle(free_machines_and_gpus)
        print('avail exps:', len(free_machines_and_gpus))
        args.free_machines_and_gpus = iter(free_machines_and_gpus)

    if len(args.exp_partition) > 0:
        args.dry_run = True

    if args.dry_run:
        free_machines_and_gpus = list(args.free_machines_and_gpus)
        from random import shuffle
        # shuffle(free_machines_and_gpus)
        i = 0
        for p in args.exp_partition:
            print()
            print('\'' + str(free_machines_and_gpus[i:i+p]) + '\'')
            i = i + p
        if i < len(free_machines_and_gpus):
            print()
            print('\'' + str(free_machines_and_gpus[i:]) + '\'')
        print()
        exit()

def get_instance_kwargs(args, num_exps, variant):
    if args.mode in [
        'newton1', 'newton3', 'newton4', 'newton6', 'newton7',
        'ada', 'alan', 'grace', 'claude',
    ]: # 'newton2', 'newton5',
        mode = 'ssh'
        ssh_host = args.mode
        if args.gpu_id is None:
            gpu_id = next(args.free_machines_and_gpus)[1]
        else:
            gpu_id = args.gpu_id
        print('using', ssh_host, gpu_id)
    elif args.mode in ['lab', 'check_lab']:
        mode = 'ssh'
        ssh_host, gpu_id = next(args.free_machines_and_gpus)
        print('using', ssh_host, gpu_id)
    else:
        mode = args.mode
        ssh_host = None
        gpu_id = args.gpu_id

    if mode == 'local_docker':
        interactive_docker = True
    else:
        interactive_docker = args.interactive_ssh

    instance_kwargs = dict(
        mode=mode,
        ssh_host=ssh_host,
        use_gpu=(not args.no_gpu),
        gpu_id=gpu_id,
        num_exps_per_instance=int(num_exps),
        interactive_docker=interactive_docker,
    )

    variant['instance_kwargs'] = instance_kwargs
    return instance_kwargs

def get_exp_prefix(args, variant, type='train-tdm'):
    if 'vae_variant' in variant:
        if 'state' in variant['vae_variant'].get('vae_type', 'VAE'):
            data_type = 'full-state'
        else:
            data_type = None
    else:
        data_type = 'full-state'

    prefix_list = [type, data_type, args.label]

    while None in prefix_list: prefix_list.remove(None)
    while '' in prefix_list: prefix_list.remove('')
    exp_prefix = '-'.join(prefix_list)

    return exp_prefix

def get_num_exps_for_instances(args):
    import numpy as np
    import math

    if args.mode == 'check_lab':
        num_exps_for_instances = np.ones(args.num_free_machines_and_gpus, dtype=np.int32)
        return num_exps_for_instances

    if args.mode == 'ec2' and (not args.no_gpu):
        max_exps_per_instance = args.max_exps_per_instance
    else:
        max_exps_per_instance = 1

    num_exps_for_instances = np.ones(int(math.ceil(args.num_seeds / max_exps_per_instance)), dtype=np.int32) \
                             * max_exps_per_instance
    num_exps_for_instances[-1] -= (np.sum(num_exps_for_instances) - args.num_seeds)

    return num_exps_for_instances

def update_snapshot_gap_and_save_period(variant):
    import math

    rl_variant = variant['rl_variant']
    if 'algo_kwargs' not in rl_variant:
        return 0

    snapshot_gap = int(math.ceil(rl_variant['algo_kwargs']['num_epochs'] / 10))

    return snapshot_gap

def query_machines(machines=None, args=None):
    if machines is None:
        machines = [
            'ada', # TESLA V100
            'alan',  # TESLA P100
            'newton1', # Titan X (pascal)
            'newton3', # Titan X (pascal)
            'newton4',  # Titan X (pascal)
            'newton6', # Titan Xp
            'newton7', # Titan Xp
            # 'grace', # GeForce GTX 1080 Ti
            'claude', # GeForce GTX 1080 Ti
        ]

        # ['newton2', # Titan X (pascal),
        #  'newton5',  # Titan X (pascal)# ]

    free_machines_info = {}

    if args is not None:
        threshold_for_free_gpu = args.threshold_for_free_gpu
    else:
        threshold_for_free_gpu = 0.05

    for machine in machines:
        cmd = 'ssh {} \'free -m\' 2>&1'.format(machine)
        results = os.popen(cmd).read()
        results = results.splitlines()
        if len(results) < 2:
            continue
        else:
            free_machines_info[machine] = {}
            results = results[1]

        free_memory_mb = [int(s) for s in results.split() if s.isdigit()][-1]
        free_memory_gb = int(free_memory_mb / 1000)
        avail_memory_gb = free_memory_gb - 8.0
        free_machines_info[machine]['mem'] = avail_memory_gb

        cmd = 'ssh {} \'nvidia-smi --query-gpu=memory.used,memory.total --format=csv\' 2>&1'.format(machine)
        results = os.popen(cmd).read()
        results = results.splitlines()[1:]
        gpu_ids = []
        gpu_free_mem = []
        for gpu_id in range(len(results)):
            gpu_info = results[gpu_id]
            used, total = gpu_info.split(',')
            used, total = int(used[:-4]), int(total[:-4])
            free_mem = total - used
            if used / total <= threshold_for_free_gpu:
                # if machine == 'newton4' and gpu_id == 2: ### this gpu is broken ###
                #     continue
                if machine == 'newton5':
                    gpu_ids.append((gpu_id + 1) % 4)
                else:
                    gpu_ids.append(gpu_id)
                gpu_free_mem.append(free_mem)
        free_machines_info[machine]['gpu_ids'] = gpu_ids
        free_machines_info[machine]['gpu_free_mem'] = gpu_free_mem

    return free_machines_info
