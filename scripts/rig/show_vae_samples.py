import argparse
import torch
from torchvision.utils import make_grid

import rlkit.torch.pytorch_util as ptu
import pickle
import uuid
import cv2
from PIL import Image


filename = str(uuid.uuid4())


def simulate_policy(args):
    ptu.set_gpu_mode(True)
    model = pickle.load(open(args.file, "rb")) # joblib.load(args.file)
    model.to(ptu.device)
    import ipdb; ipdb.set_trace()
    samples = ptu.Variable(torch.randn(64, model.representation_size))
    samples = model.decode(samples).cpu()
    # for sample in samples:
    #     tensor = sample.data.view(64, model.input_channels, model.imsize, model.imsize)
    #     tensor = tensor.cpu()
    #     img = ptu.get_numpy(tensor)
    #     cv2.imshow('img', img.reshape(3, 84, 84).transpose())
    #     cv2.waitKey(1)

    tensor = samples.data.view(64, model.input_channels, model.imsize,
                              model.imsize)
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=8)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.show()
    # cv2.imshow('img', im)
    # cv2.waitKey(1)
    # for sample in samples:
    #     tensor = tensor.cpu()
    #     img = ptu.get_numpy(tensor)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
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
