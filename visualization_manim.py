import pickle
from manim import *
import numpy as np


class AnimationImages(Scene):
    def construct(self):
        with open("n_values", "rb") as file:
            n_values = pickle.load(file)[900:]

        n = 256 / 1.25
        images = []

        for table in n_values:

            imageArray = np.uint8(
                [[int(np.clip(table[i, j] * n, 0, 256)) for i in range(table.shape[0])] for j in range(table.shape[1])]
            )
            image = ImageMobject(imageArray).scale(100)
            # image.background_rectangle = SurroundingRectangle(image, GREEN)

            images.append(image)

        self.add(images[0])

        for i in range(len(images)-1):
            self.play(Transform(images[i], images[i+1]), run_time=0.01)
