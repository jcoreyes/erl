import numpy as np
from PIL import Image


class Renderer(object):
    def __init__(
            self,
            img_width=48,
            img_height=48,
            init_camera=None,
            grayscale=False,
            normalize=True,
            flatten=False,
            input_image_format='HWC',
            output_image_format='HWC',
    ):
        """Render an image."""
        if input_image_format not in {'HWC', 'CWH'}:
            raise ValueError(
                "Only support input format of WHC or CWH, not {}".format(
                    input_image_format
                )
            )
        if output_image_format not in {'HWC', 'CWH'}:
            raise ValueError(
                "Only support output format of WHC or CWH, not {}".format(
                    output_image_format
                )
            )
        self.img_width = img_width
        self.img_height = img_height
        self.init_camera = init_camera
        self.grayscale = grayscale
        self.normalize = normalize
        self.flatten = flatten
        self.channels = 1 if grayscale else 3
        self.input_image_format = input_image_format
        self.output_image_format = output_image_format

        self._camera_is_initialized = False

    def create_image(self, env):
        if not self._camera_is_initialized and self.init_camera is not None:
            env.initialize_camera(self.init_camera)
            self._camera_is_initialized = True

        image_obs = env.get_image(
            width=self.img_width,
            height=self.img_height,
        )
        if self.grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self.normalize:
            image_obs = image_obs / 255.0
        if self.input_image_format != self.output_image_format:
            image_obs = image_obs.transpose()
        assert image_obs.shape == self.image_shape
        if self.flatten:
            return image_obs.flatten()
        else:
            return image_obs

    @property
    def image_shape(self):
        if self.output_image_format == 'HWC':
            return self.img_height, self.img_width, self.channels
        elif self.output_image_format == 'CWH':
            return self.channels, self.img_width, self.img_height
        else:
            raise ValueError(self.output_image_format)
