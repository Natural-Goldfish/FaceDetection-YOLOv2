import json
import copy
import os
from src.data_argumentation import *
from torch.utils.data import Dataset


class FDDBDataset(Dataset):
    def __init__(self, root_path = "data", mode = "train", image_size = 416, is_training = True):
        if mode in ["train", "val"] :
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
        objects = copy.deepcopy(self.anno_file[idx]["Coordinate"])
        
        for i in range(len(objects)):
            objects[i]["Class"] = [0 for j in range(len(self.class_name))]
            objects[i]["Class"][self.class_name.index(self.anno_file[i]["Object_class"])] = 1

        transforms = Resize(self.image_size)
        image, objects = transforms((image, objects))
        
        return np.transpose(np.array(image, dtype = np.float32), (2, 0 ,1)), objects

if __name__ == '__main__':
    dataset = FDDBDataset()
    image, objects = dataset[0]
    print(image.shape)
    
    