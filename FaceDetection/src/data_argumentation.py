import numpy as np
import cv2
import torch

class Resize(object):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        width_ratio = float(self.image_size) / width
        height_ratio = float(self.image_size) / height
        new_label = []

        for lb in label:
            resized_xmin = lb["Xmin"] * width_ratio
            resized_ymin = lb["Ymin"] * height_ratio
            resized_xmax = lb["Xmax"] * width_ratio
            resized_ymax = lb["Ymax"] * height_ratio
            class_idx = lb["Class"]
            resize_width = resized_xmax - resized_xmin
            resize_height = resized_ymax - resized_ymin
            new_label.append([resized_xmin, resized_ymin, resize_width, resize_height, class_idx])

        return image, new_label