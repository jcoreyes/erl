import numpy as np
import argparse
import sys
import os
import json
import math
import itertools
import csv

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def unique(l):
    return list(set(l))

def flatten(l):
    return [item for sublist in l for item in sublist]

def load_params(params_json_path):
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data

def flatten_dict(d):
    flat_params = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in flatten_dict(v).items():
                flat_params[k + "." + subk] = subv
        else:
            flat_params[k] = v
    return flat_params

def load_exps_data(
        exp_folder_paths,
        data_filename='progress.csv',
        params_filename='params.json',
        disable_variant=False,
):
    exps = []
    for exp_folder_path in exp_folder_paths:
        exps += [x[0] for x in os.walk(exp_folder_path)]
    exps_data = []
    params_filename='params.json'
    for exp in exps:
        try:
            exp_path = exp
            params_json_path = os.path.join(exp_path, params_filename)
            variant_json_path = os.path.join(exp_path, "variant.json")
            progress_csv_path = os.path.join(exp_path, data_filename)
            if os.stat(progress_csv_path).st_size == 0:
                progress_csv_path = os.path.join(exp_path, "log.txt")
            progress = load_progress(progress_csv_path)
            if disable_variant:
                params = load_params(params_json_path)
            else:
                try:
                    params = load_params(variant_json_path)
                except IOError:
                    params = load_params(params_json_path)
            exps_data.append(AttrDict(
                progress=progress,
                params=params,
                flat_params=flatten_dict(params)))
        except IOError as e:
            print(e)
    return exps_data

def load_progress(progress_csv_path):
    print("Reading %s" % progress_csv_path)
    entries = dict()
    if progress_csv_path.split('.')[-1] == "csv":
        delimiter = ','
    else:
        delimiter = '\t'
    with open(progress_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            for k, v in row.items():
                if k not in entries:
                    entries[k] = []
                try:
                    entries[k].append(float(v))
                except:
                    entries[k].append(0.)
    entries = dict([(k, np.array(v)) for k, v in entries.items()])
    return entries

def extract_distinct_params(exps_data, excluded_params=('seed', 'log_dir'), l=1):

    try:
        params_as_evalable_strings = [
            list(
                map(
                    smart_repr,
                    list(d.flat_params.items())
                )
            )
            for d in exps_data
        ]
        unique_params = unique(
            flatten(
                params_as_evalable_strings
            )
        )
        stringified_pairs = sorted(
            map(
                smart_eval,
                unique_params
            ),
            key=lambda x: (
                tuple(smart_repr(i) for i in x)
                # tuple(0. if it is None else it for it in x),
            )
        )
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()
    proposals = [(k, [x[1] for x in v])
                 for k, v in itertools.groupby(stringified_pairs, lambda x: x[0])]
    filtered = [
        (k, v) for (k, v) in proposals
        if k == 'version' or (
            len(v) > l and all(
                [k.find(excluded_param) != 0
                 for excluded_param in excluded_params]
            )
        )
    ]
    return filtered

def smart_repr(x):
    if isinstance(x, tuple):
        if len(x) == 0:
            return "tuple()"
        elif len(x) == 1:
            return "(%s,)" % smart_repr(x[0])
        else:
            return "(" + ",".join(map(smart_repr, x)) + ")"
    elif isinstance(x, list):
        if len(x) == 0:
            return "[]"
        elif len(x) == 1:
            return "[%s,]" % smart_repr(x[0])
        else:
            return "[" + ",".join(map(smart_repr, x)) + "]"
    else:
        if hasattr(x, "__call__"):
            return "__import__('pydoc').locate('%s')" % (x.__module__ + "." + x.__name__)
        elif isinstance(x, float) and math.isnan(x):
            return 'float("nan")'
        else:
            return repr(x)


def smart_eval(string):
    string = string.replace(',inf)', ',"inf")')
    return eval(string)


def reload_data():
    global exps_data
    global plottable_keys
    global distinct_params
    exps_data = load_exps_data(
        args.data_paths,
        args.data_filename,
       # args.params_filename,
        args.disable_variant,
    )
    plottable_keys = list(
        set(flatten(list(exp.progress.keys()) for exp in exps_data)))
    plottable_keys = sorted([k for k in plottable_keys if k is not None])
    distinct_params = sorted(extract_distinct_params(exps_data))

    max_info = {}

    for exp in exps_data:
        env = exp['params']['env']
        mod_env_params = exp['params'].get('env_mod', dict(gravity=1, friction=1, ctrlrange=1, gear=1))
        score = exp['progress']['eval/Average Returns'][-1]
        if env not in max_info or score > max_info[env]['score']:
            max_info[env] = dict(score=score, mod_env_params=mod_env_params)
    for k, v in sorted(max_info.items()):
        print(k, v)


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs='*')
    parser.add_argument("--prefix", type=str, nargs='?', default="???")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--disable-variant", default=False, action='store_true')
    parser.add_argument("--data-filename",
                        default='progress.csv',
                        help='name of data file.')
    parser.add_argument("--params-filename",
                        default='params.json',
                        help='name of params file.')
    args = parser.parse_args(sys.argv[1:])

    # load all folders following a prefix
    if args.prefix != "???":
        args.data_paths = []
        dirname = os.path.dirname(args.prefix)
        subdirprefix = os.path.basename(args.prefix)
        for subdirname in os.listdir(dirname):
            path = os.path.join(dirname, subdirname)
            if os.path.isdir(path) and (subdirprefix in subdirname):
                args.data_paths.append(path)
    print("Importing data from {path}...".format(path=args.data_paths))
    reload_data()

if __name__ == '__main__':
    main()