import numpy as np
from PIL import Image


class Renderer(object):
    def __init__(
            self,
            img_width=48,
            img_height=48,
            num_channels=3,
            init_camera=None,
            normalize_img=True,
            flatten_img=False,
            input_img_format='HWC',
            output_img_format='HWC',
    ):
        """Render an image."""
        if input_img_format not in {'HWC', 'CWH'}:
            raise ValueError(
                "Only support input format of WHC or CWH, not {}".format(
                    input_img_format
                )
            )
        if output_img_format not in {'HWC', 'CWH'}:
            raise ValueError(
                "Only support output format of WHC or CWH, not {}".format(
                    output_img_format
                )
            )
        self._img_width = img_width
        self._img_height = img_height
        self._init_camera = init_camera
        self._grayscale = num_channels == 1
        self._normalize_imgs = normalize_img
        self._flatten = flatten_img
        self._num_channels = num_channels
        self.input_image_format = input_img_format
        self.output_image_format = output_img_format
        self._camera_is_initialized = False

        self._camera_is_initialized = False

    def create_image(self, env):
        if not self._camera_is_initialized and self._init_camera is not None:
            env.initialize_camera(self._init_camera)
            self._camera_is_initialized = True

        image_obs = env.get_image(
            width=self._img_width,
            height=self._img_height,
        )
        if self._grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self._normalize_imgs:
            image_obs = image_obs / 255.0
        if self.input_image_format != self.output_image_format:
            image_obs = image_obs.transpose()
        assert image_obs.shape == self.image_shape
        if self._flatten:
            return image_obs.flatten()
        else:
            return image_obs

    @property
    def image_is_normalized(self):
        return self._normalize_imgs

    @property
    def image_shape(self):
        if self.output_image_format == 'HWC':
            return self._img_height, self._img_width, self._num_channels
        elif self.output_image_format == 'CWH':
            return self._num_channels, self._img_width, self._img_height
        else:
            raise ValueError(self.output_image_format)
