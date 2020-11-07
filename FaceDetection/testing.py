from src.network import *
from src.utils import draw_rec, pred_boxes, test_image_processing, iou
import os
import cv2
import numpy as np
import copy

MODEL_PATH = 'data\\models'
MODEL_NAME = "almost_check_point_150.pth"
SAMPLE_IMAGE_PATH = 'data\\image\\train'
SAMPLE_IMAGE_NAME = 'IMG_3.jpg'
NUM_CLASSES = 1

def test():
    with torch.no_grad():
        model = Yolo()
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME)))
        model.eval()

        img = cv2.imread(os.path.join(SAMPLE_IMAGE_PATH, SAMPLE_IMAGE_NAME))
        input_img = test_image_processing(img, 418)
        output = model(input_img)
        coord_box, conf_box, cls_box = pred_boxes(output, model.anchors)
        coord_box = coord_box.transpose(1, 2).contiguous().view(-1, 4)
        conf_box = conf_box.view(-1, 1, 13*13)
        confidence_score_box = cls_box * conf_box.expand_as(cls_box)
        confidence_score_box = confidence_score_box.transpose(1, 2).contiguous().view(-1, NUM_CLASSES).contiguous().view(NUM_CLASSES, -1)
        values, indices = confidence_score_box.sort(descending = True, dim = 1)
        for k in range(NUM_CLASSES):
            for i in range(13*13*5):
                if confidence_score_box[k][indices[k][i]] == 0 : continue
                for j in range(i + 1, 13*13*5):
                    iou_ = iou(coord_box[indices[k][i]].view(1, -1), coord_box[indices[k][j]].view(1, -1))
                    if (iou_ > 0.5) : confidence_score_box[k][indices[k][j]] = 0
        
        for k in range(NUM_CLASSES):
            for i in range(len(confidence_score_box[k])):
                if confidence_score_box[k][i] > 0.8:
                    xmin = coord_box[i][0] - coord_box[i][2]/2
                    xmax = coord_box[i][0] + coord_box[i][2]/2
                    ymin = coord_box[i][1] - coord_box[i][3]/2
                    ymax = coord_box[i][1] + coord_box[i][3]/2
                    img = cv2.rectangle(img, (xmin*32, ymin*32), (xmax*32, ymax*32), (255, 0, 0), 3)
        cv2.imshow("test", img)
        cv2.waitKey(0)

if __name__ == "__main__" :
    test()