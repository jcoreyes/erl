import argparse
import torch
from torchvision.utils import make_grid

import railrl.torch.pytorch_util as ptu
import pickle
import uuid
import cv2
from PIL import Image
import numpy as np
import os.path as op


filename = str(uuid.uuid4())


def simulate_policy(args):
    ptu.set_gpu_mode(True)
    model = pickle.load(open(args.file, "rb")) # joblib.load(args.file)
    model.to(ptu.device)
    imgs = np.load(args.imgfile)
    import ipdb; ipdb.set_trace()
    z = model.encode(ptu.np_to_var(imgs))
    samples = model.decode(z).cpu()

    recon_imgs = samples.data.view(64, model.input_channels, model.imsize,
                              model.imsize)
    recon_imgs = recon_imgs.cpu()
    grid = make_grid(recon_imgs, nrow=8)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.show()
    # cv2.imshow('img', im)
    # cv2.waitKey(1)
    # for sample in samples:
    #     tensor = tensor.cpu()
    #     img = ptu.get_numpy(tensor)
    comparison = torch.cat([
        recon_imgs,
        imgs,
    ])
    save_dir = osp.join(logger.get_snapshot_dir(), 'r%d.png' % epoch)
    save_image(comparison.data.cpu(), save_dir, nrow=n)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--imgfile', type=str,
                        help='path to the img (npy) file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
