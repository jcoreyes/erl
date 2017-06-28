"""
Analyze the result of a hyperparameter sweep.

Usage:
```
$ python analyze_hyperparameter_sweep.py path/to/exp_directory
```
"""
import argparse
from os.path import join

from fanova import visualizer, CategoricalHyperparameter

from railrl.misc.fanova_util import get_fanova_info
from railrl.misc.html_report import HTMLReport
import railrl.misc.visualization_util as vu


def get_param_importance(f, param):
    param_name_tuple = (param.name, )
    result = f.quantify_importance(param_name_tuple)
    return result[param_name_tuple]['total importance']


def generate_report(fanova_info, base_dir):
    f, config_space, X, Y, param_names = fanova_info
    report = HTMLReport(
        join(base_dir, 'report.html'), images_per_row=3,
    )

    """
    List the top 10 parameters.
    """
    N = min(10, len(Y))
    best_idxs = Y.argsort()[-N:][::-1]
    for rank, i in enumerate(best_idxs):
        report.add_text("Rank {} params, with score = {}:".format(rank+1, Y[i]))
        for name, value in zip(param_names, X[i, :]):
            report.add_text("\t{} = {}\n".format(name, value))

    vis = visualizer.Visualizer(f, config_space)
    cs_params = config_space.get_hyperparameters()
    importances = [get_param_importance(f, param) for param in cs_params]
    param_and_importance = sorted(
        zip(cs_params, importances),
        key=lambda x: -x[1],
    )

    """
    Plot individual marginals.
    """
    for param, importance in param_and_importance:
        param_name = param.name
        if isinstance(param, CategoricalHyperparameter):
            vis.plot_categorical_marginal(param_name, show=False)
        else:
            vis.plot_marginal(param_name, show=False)
        img = vu.save_image()
        report.add_image(
            img,
            "Marginal for {}.\nImportance = {}".format(param_name, importance),
        )

    """
    Plot pairwise marginals.
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", help="experiment dir, e.g., /tmp/experiments")
    args = parser.parse_args()
    fanova_info = get_fanova_info(args.expdir)
    generate_report(fanova_info, args.expdir)


if __name__ == '__main__':
    main()
