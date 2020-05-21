import numpy as np
from PIL import Image

from railrl.envs.images import Renderer


class GymEnvRenderer(Renderer):
    # TODO: switch everything to this env.render interface
    def create_image(self, env):
        if not self._camera_is_initialized and self._init_camera is not None:
            env.initialize_camera(self._init_camera)
            self._camera_is_initialized = True

        image_obs = env.render(
            mode='rgb_array', width=self._img_width, height=self._img_height
        )
        if self._grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self._normalize_imgs:
            image_obs = image_obs / 255.0
        transpose_index = [self.input_image_format.index(c) for c in
                           self.output_image_format]
        image_obs = image_obs.transpose(transpose_index)
        assert image_obs.shape == self.image_shape
        if self._flatten:
            return image_obs.flatten()
        else:
            return image_obs
