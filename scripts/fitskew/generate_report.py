import argparse
import glob
import json
from os.path import join
import pickle
import re

from railrl.misc.html_report import HTMLReport
from railrl.torch.vae.skew.skewed_vae_with_histogram import (
    visualize_vae_samples, visualize_vae,
)
from railrl.torch.vae.skew.datasets import project_square_border_np_4x4


def append_itr(paths):
    """
    Convert 'itr_32.pkl' into ('itr_32.pkl', 32)
    """
    for path in paths:
        match = re.compile('itr_([0-9]*).pkl').search(path)
        if match is not None:
            yield path, int(match.group(1))


def get_key_recursive(recursive_dict, key):
    for k, v in recursive_dict.items():
        if k == key:
            return v
        if isinstance(v, dict):
            child_result = get_key_recursive(v, key)
            if child_result is not None:
                return child_result
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--report_name', type=str,
                        default='report_retroactive.html')

    args = parser.parse_args()
    dir = args.dir
    report_name = args.report_name
    # dir = '/home/vitchyr/git/railrl/data/doodads3/10-30-point2d-debug-why-alpha0-crashes/10-30-point2d-debug-why-alpha0-crashes_2018_10_30_19_57_53_id000--s54973/'

    with open(join(dir, 'variant.json')) as variant_file:
        variant = json.load(variant_file)
    skew_config = get_key_recursive(variant, 'skew_config')
    pkl_paths = glob.glob(dir + '/*.pkl')
    numbered_paths = append_itr(pkl_paths)
    ordered_numbered_paths = sorted(numbered_paths, key=lambda x: x[1])

    report = HTMLReport(join(dir, report_name), images_per_row=5)

    for path, itr in ordered_numbered_paths:
        if 30 > itr or itr > 35:
            continue
        print("Processing itration {}".format(itr))
        snapshot = pickle.load(open(path, "rb"))
        vae = snapshot['p_theta']
        vae.to('cpu')
        vae_train_data = snapshot['train_data']
        dynamics = snapshot.get('dynamics', project_square_border_np_4x4)
        report.add_header("Iteration {}".format(itr))
        visualize_vae_samples(
            itr,
            vae_train_data,
            vae,
            report,
            dynamics=dynamics,
        )
        visualize_vae(
            vae,
            skew_config,
            report,
            title="Post-skew",
        )

    report.save()
    print("Report saved to")
    print(report.path)



if __name__ == '__main__':
    main()