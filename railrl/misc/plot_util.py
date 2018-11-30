import matplotlib.pyplot as plt
import numpy as np

import json
from pprint import pprint
try:
    import rllab.viskit.core as core
except:
    import viskit.core as core

read_tb = lambda: None
import glob
import os
import itertools

from contextlib import contextmanager
import sys, os

true_fn = lambda p: True
identity_fn = lambda x: x

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def load_exps(dirnames, filter_fn=true_fn, suppress_output=False):
    if suppress_output:
        with suppress_stdout():
            exps = core.load_exps_data(dirnames)
    else:
        exps = core.load_exps_data(dirnames)
    good_exps = []
    for e in exps:
        if filter_fn(e):
            good_exps.append(e)
    return good_exps

def tag_exps(exps, tag_key, tag_value):
    for e in exps:
        e["flat_params"][tag_key] = tag_value

def read_params_from_output(filename, maxlines=200):
    if not filename in cached_params:
        f = open(filename, "r")
        params = {}
        for i in range(maxlines):
            l = f.readline()
            if not ":" in l:
                break
            kv = l[l.find("]")+1:]
            colon = kv.find(":")
            k, v = kv[:colon], kv[colon+1:]
            params[k.strip()] = v.strip()
        f.close()
        cached_params[filename] = params
    return cached_params[filename]

def prettify(p, key):
    """Postprocessing p[key] for printing"""
    return p[key]

def prettify_configuration(config):
    if not config:
        return ""
    s = ""
    for c in config:
        k, v = str(c[0]), str(c[1])
        x = ""
        x = k + "=" + v + ", "
        s += x
    return s[:-2]

def to_array(lists):
    """Converts lists of different lengths into a left-aligned 2D array"""
    M = len(lists)
    N = max(len(y) for y in lists)
    output = np.zeros((M, N))
    output[:] = np.nan
    for i in range(M):
        y = lists[i]
        n = len(y)
        output[i, :n] = y
    return output

def filter_by_flat_params(d):
    def f(l):
        for k in d:
            if l['flat_params'][k] != d[k]:
                return False
        return True
    return f

def comparison(exps, key, vary = ["expdir"], f=true_fn, smooth=identity_fn, figsize=(5, 3.5),
    xlabel="Number of env steps total", default_vary=False, xlim=None, ylim=None,
    print_final=False, print_max=False, print_min=False, print_plot=True,
    reduce_op=sum, method_order=None, remap_keys={},
    label_to_color=None,
):
    """exps is result of core.load_exps_data
    key is (what we might think is) the effect variable
    vary is (what we might think is) the causal variable
    f is a filter function on the exp parameters"""
    if print_plot:
        plt.figure(figsize=figsize)
        plt.title("Vary " + " ".join(vary))
        plt.ylabel(key)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

    y_data = {}
    x_data = {}
    def lookup(v):
        if v in l['flat_params']:
            return str(l['flat_params'][v])
        if v in default_vary:
            return str(default_vary[v])
        print(v)
        error_key_not_found_in_flat_params
    for l in exps:
        if f(l) and l['progress']:
            label = " ".join([v + ":" + lookup(v) for v in vary])
            ys = y_data.setdefault(label, [])
            xs = x_data.setdefault(label, [])

            d = l['progress']
            x = d[xlabel]
            y = [0]
            if type(key) is list:
                vals = []
                for k in key:
                    if k in d:
                        vals.append(d[k])
                    elif k in remap_keys:
                        k_new = remap_keys[k]
                        vals.append(d[k_new])
                    else:
                        error_key_not_found_in_logs
                y = reduce_op(vals)
            else:
                if key in d:
                    y = d[key]
                else:
                    print("not found", key)
                    print(d.keys())

            y_smooth = smooth(y)
            x_smooth = x[-len(y_smooth):]
            ys.append(y_smooth)
            xs.append(x_smooth)
            # print(x_smooth.shape, y_smooth.shape)

    lines = []
    labels = sorted(y_data.keys())
    if method_order:
        labels = np.array(labels)[np.array(method_order)]
    for label in labels:
        ys = to_array(y_data[label])
        x = np.nanmean(to_array(x_data[label]), axis=0)
        y = np.nanmean(ys, axis=0)
        if x.shape != y.shape:
            print("label shape mismatch:", x.shape, y.shape)
            continue

        s = np.nanstd(ys, axis=0) / (len(ys) ** 0.5)
        if print_plot:
            if label_to_color is None:
                plt.fill_between(x, y-1.96*s, y+1.96*s, alpha=0.2)
                line, = plt.plot(x, y, label=str(label))
            else:
                label_without_vary_prefix = label.split(":")[-1]
                color = label_to_color[label_without_vary_prefix]
                plt.fill_between(x, y-1.96*s, y+1.96*s, alpha=0.2, color=color)
                line, = plt.plot(x, y, label=str(label), color=color)
            lines.append(line)

        if print_final:
            print(label, y[-1])
        snapshot = 20
        if len(y) > snapshot:
            if print_max:
                i = np.argmax(y[::snapshot]) * snapshot
                print(label, i, y[i])
            if print_min:
                i = np.argmin(y[::snapshot]) * snapshot
                print(label, i, y[i])

    if print_plot:
        plt.legend(handles=lines, bbox_to_anchor=(1.5, 0.75))

    return lines

def split(exps,
    keys,
    vary = "expdir",
    split=[],
    f=true_fn,
    w="evaluator",
    smooth=identity_fn,
    figsize=(5, 3),
    suppress_output=False,
    xlabel="Number of env steps total",
    default_vary=False,
    xlim=None, ylim=None,
    print_final=False, print_max=False, print_min=False, print_plot=True):
    split_values = {}
    for s in split:
        split_values[s] = set()
    for l in exps:
        if f(l):
            for s in split:
                split_values[s].add(l['flat_params'][s])
    print(split_values)

    configurations = []
    for s in split_values:
        c = []
        for v in split_values[s]:
            c.append((s, v))
        configurations.append(c)
    for c in itertools.product(*configurations):
        fsplit = lambda exp: all([exp['flat_params'][k] == v for k, v in c]) and f(exp)
        # for exp in exps:
        #     print(fsplit(exp), exp['flat_params'])
        for key in keys:
            if print_final or print_max or print_min:
                print(key, c)
            comparison(exps, key, vary, f=fsplit, smooth=smooth,
                figsize=figsize, xlabel=xlabel, default_vary=default_vary, xlim=xlim, ylim=ylim,
                print_final=print_final, print_max=print_max, print_min=print_min, print_plot=print_plot)
            if print_plot:
                plt.title(prettify_configuration(c) + " Vary " + " ".join(vary))


def min_length(trials, key):
    min_len = np.inf
    for trial in trials:
        values_ts = trial.data[key]
        min_len = min(min_len, len(values_ts))
    return min_len


def plot_trials(
        name_to_trials,
        y_keys=None,
        x_key="Number of env steps total",
        x_label=None,
        y_label=None,
        process_values=sum,
        process_time_series=identity_fn,
):
    if isinstance(y_keys, str):
        y_keys = [y_keys]
    if x_label is None:
        x_label = x_key
    if y_label is None:
        y_label = "+".join(y_keys)
    y_keys = [y.replace(" ", "_") for y in y_keys]
    x_key = x_key.replace(" ", "_")
    all_trials = [t for trials in name_to_trials.values() for t in trials]
    min_len = min_length(all_trials, x_key)
    for name, trials in name_to_trials.items():
        all_values = []
        for trial in trials:
            if len(y_keys) == 1:
                values = trial.data[y_keys[0]]
            else:
                multiple_values = [
                    trial.data[k] for k in y_keys
                ]
                values = process_values(multiple_values)
            values = process_time_series(values)
            all_values.append(values)
            x_values = trial.data[x_key][:min_len]
        try:
            y_values = np.vstack([values[:min_len] for values in all_values])
        except ValueError as e:
            import ipdb; ipdb.set_trace()
            print(e)
        mean = np.mean(y_values, axis=0)
        std = np.std(y_values, axis=0)
        plt.fill_between(x_values, mean - std, mean + std, alpha=0.1)
        plt.plot(x_values, mean, label=name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def ma_filter(N):
    return lambda x: moving_average(x, N)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def padded_ma_filter(N, **kwargs):
    return lambda x: padded_moving_average(x, N, **kwargs)


def padded_moving_average(data_array, window=5, avg_only_from_left=True):
    """Does not affect the length"""
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        if avg_only_from_left:
            indices = list(range(max(i - window + 1, 0), i+1))
        else:
            indices = list(range(max(i - window + 1, 0),
                                 min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


import itertools
def scatterplot_matrix(data1, data2, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = len(data1), len(data2)
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(16,16))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in itertools.product(range(numvars), range(numdata)):
        axes[i,j].scatter(data1[i], data2[j], **kwargs)
        label = "{:6.3f}".format(np.corrcoef([data1[i], data2[j]])[0, 1])
        axes[i,j].annotate(label, (0.1, 0.9), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    # for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
    #     axes[j,i].xaxis.set_visible(True)
    #     axes[i,j].yaxis.set_visible(True)

    return fig
