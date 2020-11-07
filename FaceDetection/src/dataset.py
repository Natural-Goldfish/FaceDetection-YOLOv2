import json
import copy
import os
from src.data_argumentation import *
from torch.utils.data import Dataset


class FDDBDataset(Dataset):
    def __init__(self, mode, image_size = 416, root_path = "data"):
        if mode in ["train", "test"] :
            self.image_path = os.path.join(root_path, 'image', "{}".format(mode))
            self.image_size = image_size
            self.anno_file = json.load(open(os.path.join(root_path, "annotation_{}.json".format(mode))), encoding = 'utf-8')
            self.num_images = len(self.anno_file)
        self.class_name = ["Face"]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.anno_file[idx]["Img_id"]) + ".jpg"
        image = cv2.imread(image_path)
        objects = copy.deepcopy(self.anno_file[idx]["Objects"])
        
        for i in range(len(objects)):
            objects[i]["Object_class"] = 1

        transforms = Transforms([Resize(self.image_size), Numpy2Tensor()])
        image, labels = transforms((image, objects))
        return image, labels
    
    