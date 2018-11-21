import glob
import json
import pickle
from os.path import join

import argparse
import numpy as np
import re
from skvideo.io import vwrite

from railrl.misc.html_report import HTMLReport
from railrl.misc.visualization_util import gif
from railrl.torch.vae.skew.datasets import project_square_border_np_4x4
from railrl.torch.vae.skew.skewed_vae_with_histogram import (
    visualize_vae_samples, visualize_vae,
)


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
    directory = args.dir
    report_name = args.report_name

    with open(join(directory, 'variant.json')) as variant_file:
        variant = json.load(variant_file)
    skew_config = get_key_recursive(variant, 'skew_config')
    pkl_paths = glob.glob(directory + '/*.pkl')
    numbered_paths = append_itr(pkl_paths)
    ordered_numbered_paths = sorted(numbered_paths, key=lambda x: x[1])

    report = HTMLReport(join(directory, report_name), images_per_row=5)

    vae_heatmap_imgs = []
    sample_imgs = []
    for path, itr in ordered_numbered_paths:
        print("Processing iteration {}".format(itr))
        snapshot = pickle.load(open(path, "rb"))
        if 'vae' in snapshot:
            vae = snapshot['vae']
        else:
            vae = snapshot['p_theta']
        vae.to('cpu')
        vae_train_data = snapshot['train_data']
        dynamics = snapshot.get('dynamics', project_square_border_np_4x4)
        report.add_header("Iteration {}".format(itr))
        vae.xy_range = ((-4, 4), (-4, 4))
        vae_heatmap_img = visualize_vae_samples(
            itr,
            vae_train_data,
            vae,
            report,
            xlim=vae.get_plot_ranges()[0],
            ylim=vae.get_plot_ranges()[1],
            dynamics=dynamics,
        )
        sample_img = visualize_vae(
            vae,
            skew_config,
            report,
            title="Post-skew",
        )
        vae_heatmap_imgs.append(vae_heatmap_img)
        sample_imgs.append(sample_img)

    report.add_header("Summary GIFs")
    for filename, imgs in [
        ("vae_heatmaps", vae_heatmap_imgs),
        ("samples", sample_imgs),
    ]:
        video = np.stack(imgs)
        vwrite(
            '{}/{}.mp4'.format(directory, filename),
            video,
        )
        gif_file_path = '{}/{}.gif'.format(directory, filename)
        gif(gif_file_path, video)
        report.add_image(gif_file_path, txt=filename, is_url=True)

    report.save()
    print("Report saved to")
    print(report.path)



if __name__ == '__main__':
    main()