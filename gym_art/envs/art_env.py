import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
from PIL import Image, ImageDraw
import numpy as np
import random
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.spatial.distance import euclidean

def image_to_grayscale(image):
    image = color.rgb2gray(image)
    return image

class ArtEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, image_width=128, image_height=128, brush_widths=4, source="cifar10", renderer_scale=8, max_steps=100):

        if source == "mnist":
            self.image_width = 28
            self.image_height = 28

            from keras.datasets import mnist
            (self.images, _), _ =  mnist.load_data()
            self.images = self.images.astype("float32") / 255.0
        elif source == "cifar10":
            self.image_width = 32
            self.image_height = 32

            from keras.datasets import cifar10
            (self.images, _), _ =  cifar10.load_data()
        else:
            self.image_width = image_width
            self.image_height = image_height
            assert False, "Implement!"

        self.brush_widths = brush_widths
        self.renderer_scale = renderer_scale
        self.max_steps = max_steps

        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=255, shape=(self.image_width, self.image_height), dtype=np.uint8), # Target image.
            spaces.Box(low=0, high=255, shape=(self.image_width, self.image_height), dtype=np.uint8), # Canvas image.
        ])

        self.action_space =  spaces.Tuple([
            spaces.Discrete(self.image_width), # Where to paint - x.
            spaces.Discrete(self.image_height), # Where to paint - y.
            spaces.Discrete(self.brush_widths), # What to paint width.
            spaces.Discrete(255), # What to paint - intensities.
        ])

        self.screen_width = self.image_width * 2 * renderer_scale
        self.screen_height = self.image_height * renderer_scale

        self._pygame_screen = None


    def step(self, action):

        assert self._reset == True, "Did you reset the environment?"

        if self.current_step == self.max_steps:
            observation = self._get_observation()
            reward = self._get_reward()
            done = True
            info = {}
            return observation, reward, done, info

        else:
            self.current_step += 1

            action_x = action[0]
            action_y = action[1]
            action_brush_width = action[2] + 1
            action_brush_intensity = action[3]

            # Draw only non zero widths.
            #if action_brush_width != 0:
            ellipse_x1 = action_x
            ellipse_y1 = action_y
            ellipse_x2 = action_x + action_brush_width
            ellipse_y2 = action_y + action_brush_width
            ellipse = (ellipse_x1, ellipse_y1, ellipse_x2, ellipse_y2)
            self._pil_draw.ellipse(ellipse, fill=(action_brush_intensity,))
            self._image = np.array(self._pil_image)

            observation = self._get_observation()
            reward = self._get_reward()
            done = False
            info = {}
            return observation, reward, done, info


    def reset(self):
        self._reset = True

        # Select a target image.
        target_image = image_to_grayscale(random.choice(self.images))
        target_image *= 255.0
        target_image = target_image.astype("uint8")
        self._target_image = target_image
        self._target_pil_image = Image.fromarray(self._target_image)

        # Set the current image.
        self._pil_image = Image.new("L", (self.image_width, self.image_height))
        self._pil_draw = ImageDraw.Draw(self._pil_image)
        self._image = np.array(self._pil_image)


        self.current_step = 0

        # Return the observation.
        observation = self._get_observation()
        return observation


    def render(self, mode='human', close=False):
        assert self._reset == True, "Did you reset the environment?"

        # Lazy loading pygame.
        if self._pygame_screen == None:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # Consume events.
        for event in pygame.event.get():
            pass

        target_size = (self.image_width * self.renderer_scale, self.image_height * self.renderer_scale)

        # Render target image.
        image = self._target_pil_image
        image = image.resize(target_size, Image.NEAREST)
        image_string = image.convert("RGB").tobytes("raw", "RGB")
        pil_surface = pygame.image.frombuffer(image_string, target_size, "RGB")
        self._pygame_screen.blit(pil_surface, (0, 0))

        # Render image.
        image = self._pil_image
        image = image.resize(target_size, Image.NEAREST)
        image_string = image.convert("RGB").tobytes("raw", "RGB")
        pil_surface = pygame.image.frombuffer(image_string, target_size, "RGB")
        self._pygame_screen.blit(pil_surface, (self.image_width * self.renderer_scale, 0))

        # Flip
        pygame.display.flip()


    def _get_observation(self):
        assert self._target_image.shape == self._image.shape
        assert self._target_image.dtype == self._image.dtype

        # Convert to ANN-friendly range.
        observation_image_target = self._target_image / 255.0
        observation_image_target = observation_image_target.astype("float32")
        observation_image = self._image / 255.0
        observation_image = observation_image.astype("float32")

        return observation_image_target, observation_image


    def _get_reward(self):
        # Compute the absolute differences of pixel-differences.
        #print(np.min(self._image), np.max(self._image), "  ")
        #print(np.min(self._target_image), np.max(self._target_image), "  ")
        differences = self._target_image.flatten() / 255.0 - self._image.flatten() / 255.0
        absolute_differences = np.abs(differences)


        #target_non_black = 0
        #match_count_target_non_black = 0
        #for target_pixel, image_pixel in zip(self._target_image.flatten(), self._image.flatten()):
        #    target_pixel /= 255.0
        #    image_pixel /= 255.0
        #    if target_pixel != 0.0:
        #        target_non_black += 1
        #    if (np.abs(target_pixel - image_pixel) <= 0.1) and (target_pixel != 0.0):
        #        match_count_target_non_black += 1
        #reward = int(200.0 * (match_count_target_non_black / target_non_black - 0.5))

        # Count how many pixels are below threshold.
        num_pixels_below_threshold = len(absolute_differences[absolute_differences <= 0.0])
        num_pixels_below_threshold_normalized = num_pixels_below_threshold / (self.image_width * self.image_height)

        reward = int(100 * num_pixels_below_threshold_normalized) - 90
        #print(num_pixels_below_threshold, reward, end=" ")

        # Compute reward.
        #if num_pixels_below_threshold_normalized == 1.0:
        #    reward = 20.0
        #elif num_pixels_below_threshold_normalized > 0.8:
        #    reward = 10.0
        #elif num_pixels_below_threshold_normalized > 0.6:
        #    reward = 5.0
        #elif num_pixels_below_threshold_normalized > 0.4:
        #    reward = 2.0
        #elif num_pixels_below_threshold_normalized > 0.2:
        #    reward = 1.0
        #else:
        #     reward = -1.0

        return reward
