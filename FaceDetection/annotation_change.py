import os
import cv2
import numpy as np
import json
from collections import OrderedDict

class AnnotationChange(object):
    def __init__(self, txt_dir = "FDDB\\FDDB-folds", img_dir = "FDDB", new_img_dir = "data\\image"):
        self.txt_dir = txt_dir
        self.img_dir = img_dir
        self.new_img_dir = new_img_dir
        self._directory_path()
        self.json_list = []

    def _directory_path(self):
        """
        If directories for saving images and annotations, make it
        """
        if not os.path.isdir(self.new_img_dir) : os.mkdir(self.new_img_dir)
        if not os.path.isdir(os.path.join(self.new_img_dir, "train")) : os.mkdir(os.path.join(self.new_img_dir, "train"))
        if not os.path.isdir(os.path.join(self.new_img_dir, "test")) : os.mkdir(os.path.join(self.new_img_dir, "test"))

    def _save_annotation(self, purpose):
        """
        Save annotaion with different purpose
        """
        with open('data\\annotation_{}.json'.format(purpose), 'w', encoding='utf-8') as make_file:
            json.dump(self.json_list, make_file, ensure_ascii= False, indent= 4)
        self.json_list.clear()

    def __call__(self):
        """
        This changes FDDB annotations to the Yolo annotation format
        """
        train_number = 1
        test_number = 1
        for cur_file in os.listdir(self.txt_dir):
            if(cur_file.split(".")[0].split("-")[-1] == "ellipseList"):
                with open(os.path.join(self.txt_dir, cur_file), 'r') as txt_file:
                    while(True):
                        line = txt_file.readline().rstrip("\n")
                        if not line : break

                        # Image load
                        img_folder_ = line.split("/") 
                        img_dir_ = \
                            self.img_dir + "\\" + img_folder_[0] + "\\" + img_folder_[1] + "\\" + img_folder_[2] + "\\" + img_folder_[3] + "\\" + img_folder_[4] + ".jpg"
                        img = cv2.imread(img_dir_, cv2.IMREAD_COLOR)
                        
                        if(train_number > 2500) :
                            image_name = "IMG_{}.jpg".format(test_number)
                            cv2.imwrite(os.path.join(self.new_img_dir, "test", image_name), img)
                            test_number = test_number + 1

                        else :
                            image_name = "IMG_{}.jpg".format(train_number)
                            cv2.imwrite(os.path.join(self.new_img_dir, "train", image_name), img)
                            train_number = train_number + 1
                        
                        object_num = int(txt_file.readline().rstrip("\n"))
                        file_data = OrderedDict()
                        object_data = OrderedDict()
                        file_data["Img_id"] = image_name.split('.')[0]
                        file_data["Objects"] = []

                        #Do "for" operation as much as the number of objects
                        for i in range(object_num):
                            coord_ = txt_file.readline().split(" ")
                            rad = abs(float(coord_[2]))
                            h = float(coord_[0])*np.sin(rad)
                            w = float(coord_[1])*np.sin(rad)
                            object_data["Object_class"] = "Face"
                            object_data["Coordinate"] = { "Xmin" : float(coord_[3]) - w, "Ymin" : float(coord_[4]) - h, "Xmax" : float(coord_[3]) + w, "Ymax" : float(coord_[4]) + h }
                            file_data["Objects"].append(object_data)
                        self.json_list.append(file_data)

                        # Save train annotation
                        if(train_number == 2501):
                            self._save_annotation("train")
                            train_number = train_number + 1
        self._save_annotation("test")

if __name__ == "__main__":
    anno_change = AnnotationChange()
    anno_change()