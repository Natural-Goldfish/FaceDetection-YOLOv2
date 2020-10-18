import os
import cv2
import numpy as np
import json
from collections import OrderedDict

txt_dir = "FDDB\\FDDB-folds"
img_dir = "FDDB"
new_img_dir = "data\\image"

if not os.path.isdir(new_img_dir):
    os.mkdir(new_img_dir)

if not os.path.isdir(os.path.join(new_img_dir, "train")):
    os.mkdir(os.path.join(new_img_dir, "train"))

if not os.path.isdir(os.path.join(new_img_dir, "val")):
    os.mkdir(os.path.join(new_img_dir, "val"))

json_list = []
number = 1
val_number = 1

for file_ in os.listdir(txt_dir):
    if(file_.split(".")[0].split("-")[-1] == "ellipseList"):
        with open(os.path.join(txt_dir, file_), 'r') as txt_file:
            while(True):

                line = txt_file.readline().rstrip("\n")
                if not line : break

                # Image load
                img_folder_ = line.split("/") 
                img_dir_ = \
                    img_dir + "\\" + img_folder_[0] + "\\" + img_folder_[1] + "\\" + img_folder_[2] + "\\" + img_folder_[3] + "\\" + img_folder_[4] + ".jpg"
                img = cv2.imread(img_dir_, cv2.IMREAD_COLOR)
                
                if(number > 2500) :
                    image_name = "IMG_{}.jpg".format(val_number)
                    cv2.imwrite(os.path.join(new_img_dir, "val", image_name), img)
                    val_number = val_number + 1

                else :
                    image_name = "IMG_{}.jpg".format(number)
                    cv2.imwrite(os.path.join(new_img_dir, "train", image_name), img)
                    number = number + 1
                
                object_num = int(txt_file.readline().rstrip("\n"))
                file_data = OrderedDict()
                file_data["Img_id"] = image_name.split('.')[0]
                file_data["Object_class"] = 'Face'
                file_data["Coordinate"] = []

                #Do "for" operation as much as the number of objects
                for i in range(object_num):
                    
                    coord_ = txt_file.readline().split(" ")
                    rad = abs(float(coord_[2]))
                    h = float(coord_[0])*np.sin(rad)
                    w = float(coord_[1])*np.sin(rad)
                    file_data["Coordinate"].append({ "Xmin" : float(coord_[3]) - w, "Ymin" : float(coord_[4]) - h, "Xmax" : float(coord_[3]) + w, "Ymax" : float(coord_[4]) + h })
                
                json_list.append(file_data)
                if(number == 2501):
                    with open("data\\annotation_train.json", 'w', encoding = 'utf-8') as make_file:
                        json.dump(json_list, make_file, ensure_ascii = False, indent = 4)
                    json_list.clear()
                    number = number + 1

with open('data\\annotation_val.json', 'w', encoding='utf-8') as make_file:
    json.dump(json_list, make_file, ensure_ascii= False, indent= 4)