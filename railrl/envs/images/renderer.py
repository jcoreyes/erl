import numpy as np
from PIL import Image


class Renderer(object):
    def __init__(
            self,
            img_width=48,
            img_height=48,
            init_camera=None,
            transpose=True,
            grayscale=False,
            normalize=True,
            flatten=False,
    ):
        """Render an image."""
        self.img_width = img_width
        self.img_height = img_height
        self.init_camera = init_camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize
        self.flatten = flatten
        self.channels = 1 if grayscale else 3

        self.initialize_camera = False

    def create_image(self, env):
        if not self.initialize_camera and self.init_camera is not None:
            env.initialize_camera(self.init_camera)
            self.initialize_camera = True

        image_obs = env.get_image(
            width=self.img_width,
            height=self.img_height,
        )
        if self.grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self.normalize:
            image_obs = image_obs / 255.0
        if self.transpose:
            image_obs = image_obs.transpose()
        assert image_obs.shape[0] == self.channels
        if self.flatten:
            return image_obs.flatten()
        else:
            return image_obs

    @property
    def image_shape(self):
        return self.img_width, self.img_height, self.channels
