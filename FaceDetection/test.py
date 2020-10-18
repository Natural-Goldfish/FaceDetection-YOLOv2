from src import network
from src.utils import draw_rec, nms, pred_boxes
import torch
import cv2
import numpy as np
import copy

class arguments():
    def __init__(self):

        self.dataset_path = 'data/'
        self.mode = 'train'
        self.image_size = 418
        self.batch_size = 10
        self.coord_scale = 5
        self.noobj_scale = 0.5
        self.epochs = 100
        self.sample_path = 'data\\image\\val\\IMG_30.jpg'
        self.image_size = 418


def face_detect_test(args):
    model = network.Yolo()
    model.load_state_dict(torch.load("models\\almost_check_point_150.pth"))
    model.eval()
    img = cv2.imread(args.sample_path)

    img_ = cv2.resize(img, (args.image_size, args.image_size))
    draw_img = copy.deepcopy(img_)
    img_ = np.transpose(np.array(img_, dtype = np.float32), (2, 0 ,1))
    img_ = torch.tensor(img_).clone().detach().requires_grad_(False)
    img_ = img_.unsqueeze(0)


    output= model(img_)
    min_x, min_y, max_x, max_y = nms(pred_boxes(output, anchors = model.anchors))
    draw_rec(draw_img, min_x, min_y, max_x, max_y)
    
    
if __name__ == "__main__" :
    args = arguments()
    face_detect_test(args)