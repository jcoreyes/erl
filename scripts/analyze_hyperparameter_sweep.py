"""
Analyze the result of a hyperparameter sweep.

Usage:
```
$ python analyze_hyperparameter_sweep.py path/to/exp_directory
```
"""
import argparse
import os.path as osp
import subprocess

import numpy as np

from fanova import visualizer, CategoricalHyperparameter

from railrl.misc.fanova_util import get_fanova_info, FanovaInfo
from railrl.misc.html_report import HTMLReport
import railrl.misc.visualization_util as vu
from rllab.misc.instrument import query_yes_no


def get_param_importance(f, param):
    param_name_tuple = (param.name, )
    result = f.quantify_importance(param_name_tuple)
    return result[param_name_tuple]['total importance']


def max_min_mean_error(x):
    delta = x.max() - x.min()
    uniform_mean_estimate = (x.max() + x.min()) / 2
    return np.abs((x.mean() - uniform_mean_estimate) / delta)


def is_data_log_uniformly_distributed(x):
    bin_sizes, *_ = np.histogram(x)
    is_decreasing = [
        bin_sizes[i] > bin_sizes[i+1] for i in range(len(bin_sizes) - 1)
    ]
    deltas = [
        bin_sizes[i] - bin_sizes[i+1] for i in range(len(bin_sizes) - 1)
    ]
    uniform_error = max_min_mean_error(x)
    log_uniform_error = max_min_mean_error(np.log(x))
    heuristics = [
        np.mean(is_decreasing) > 0.7,
        log_uniform_error + 0.1 < uniform_error,
        bin_sizes[0] > np.mean(bin_sizes) * 1.5,
        np.mean(deltas) > 0,
    ]
    return np.mean(heuristics) >= 0.5


def generate_report(fanova_info: FanovaInfo, base_dir, param_name_to_log=None):
    if param_name_to_log is None:
        param_name_to_log = {}
    f, config_space, X, Y = fanova_info
    report = HTMLReport(
        osp.join(base_dir, 'report.html'), images_per_row=3,
    )
    param_names = [p.name for p in config_space.get_hyperparameters()]

    """
    List the top 10 parameters.
    """
    N = min(10, len(Y))
    Y[np.isnan(Y)] = np.nanmin(Y) - 1
    best_idxs = Y.argsort()[-N:][::-1]
    for rank, i in enumerate(best_idxs):
        report.add_text("Rank {} params, with score = {}:".format(rank+1, Y[i]))
        for name, value in zip(param_names, X[i, :]):
            report.add_text("\t{} = {}\n".format(name, value))

    vis = visualizer.Visualizer(f, config_space)
    cs_params = config_space.get_hyperparameters()
    importances = [get_param_importance(f, param) for param in cs_params]
    are_logs = [is_data_log_uniformly_distributed(X[:, i])
              for i in range(len(cs_params))]
    data = sorted(
        zip(cs_params, importances, are_logs),
        key=lambda x: -x[1],
    )

    print("Guesses for is_log")
    print("{")
    for param, _, is_log in data:
        name = param.name
        print("    '{}': {},".format(name, is_log))
    print("}")

    """
    Plot individual marginals.
    """
    print("Creating marginal plots")
    for param, importance, is_log in data:
        param_name = param.name
        if param_name in param_name_to_log:
            is_log = param_name_to_log[param_name]
        if isinstance(param, CategoricalHyperparameter):
            vis.plot_categorical_marginal(param_name, show=False)
        else:
            vis.plot_marginal(param_name, show=False, log_scale=is_log)
        img = vu.save_image()
        report.add_image(
            img,
            "Marginal for {}.\nImportance = {}".format(param_name, importance),
        )

    """
    Plot pairwise marginals.
    """
    print("Creating pairwise-marginal plots")
    num_params = len(cs_params)
    num_pairs = num_params * (num_params + 1) // 2
    pair_and_importance = (
        f.get_most_important_pairwise_marginals(num_pairs)
    )
    for combi, importance in pair_and_importance.items():
        param_names = []
        for p in combi:
            param_names.append(cs_params[p].name)
        vis.plot_pairwise_marginal(combi, show=False)
        img = vu.save_image()
        report.add_image(
            img,
            "Pairwise Marginal for {}.\nImportance = {}".format(
                param_names,
                importance,
            ),
        )
    abs_path = osp.abspath(report.path)
    print("Report saved to: {}".format(abs_path))
    open_report = query_yes_no("Open report?", default="yes")
    if open_report:
        cmd = "xdg-open {}".format(abs_path)
        print(cmd)
        subprocess.call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", help="experiment dir, e.g., /tmp/experiments")
    args = parser.parse_args()
    fanova_info = get_fanova_info(args.expdir)
    are_logs = {
    }

    generate_report(fanova_info, args.expdir, are_logs)


if __name__ == '__main__':
    main()
