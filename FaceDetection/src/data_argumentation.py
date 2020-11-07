import numpy as np
import cv2
import torch

class Transforms(object):
    def __init__(self, function_list):
        super().__init__()
        self.function_list = function_list
        
    def __call__(self, data):
        for function in self.function_list:
            data = function(data)
        image, labels = data
        return image, labels

class Numpy2Tensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        image, labels = data
        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype = torch.float32, requires_grad = False)
        labels = torch.tensor(labels, requires_grad = False)
        return image, labels

class Resize(object):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def __call__(self, data):
        image, objects = data
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        width_ratio = float(self.image_size) / width
        height_ratio = float(self.image_size) / height
        new_label = []

        for ob in objects:
            resized_xmin = ob["Coordinate"]["Xmin"] * width_ratio
            resized_ymin = ob["Coordinate"]["Ymin"] * height_ratio
            resized_xmax = ob["Coordinate"]["Xmax"] * width_ratio
            resized_ymax = ob["Coordinate"]["Ymax"] * height_ratio
            class_idx = ob["Object_class"]
            resize_width = resized_xmax - resized_xmin
            resize_height = resized_ymax - resized_ymin
            new_label.append([resized_xmin, resized_ymin, resize_width, resize_height, class_idx])

        return image, new_label