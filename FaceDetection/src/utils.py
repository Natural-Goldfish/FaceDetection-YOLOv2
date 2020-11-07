import torch
import cv2
import numpy as np
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    return items

def iou(B_boxes1, B_boxes2):
    """
    This method is getting IOU(Intersection of Union) between B_boxes1 and B_boxes2
    B_boxes1 should be Ground truth boxes and B_boxes2 should be pred_boxes or anchor boxes
    """
    Bb1x1, Bb2x1 = B_boxes1[:, 0] - B_boxes1[:, 2]/2, B_boxes2[:, 0] - B_boxes2[: ,2]/2
    Bb1y1, Bb2y1 = B_boxes1[:, 1] - B_boxes1[:, 3]/2, B_boxes2[:, 1] - B_boxes2[: ,3]/2
    Bb1x2, Bb2x2 = B_boxes1[:, 0] + B_boxes1[:, 2]/2, B_boxes2[:, 0] + B_boxes2[: ,2]/2
    Bb1y2, Bb2y2 = B_boxes1[:, 1] + B_boxes1[:, 2]/2, B_boxes2[:, 1] + B_boxes2[: ,2]/2

    inter_w = (Bb1x2.view(-1, 1).min(Bb2x2.view(1, -1)) - Bb1x1.view(-1, 1).max(Bb2x1.view(1, -1))).clamp(min = 0)
    inter_h = (Bb1y2.view(-1, 1).min(Bb2y2.view(1, -1)) - Bb1y1.view(-1, 1).max(Bb2y1.view(1, -1))).clamp(min = 0)
    InterSection = inter_w * inter_h

    area1, area2 = ((Bb1x2 - Bb1x1) * (Bb1y2 - Bb1y1)).view(-1, 1), ((Bb2x2 - Bb2x1) * (Bb2y2 - Bb2y1)).view(1, -1)
    Union = (area1 + area2) - InterSection
    return InterSection / Union

def pred_boxes(output, anchors, num_anchors = 5, num_classes = 1, num_grid = 13):
    anchors = torch.tensor(anchors).view(-1, 2)
    output = output.view(1, num_anchors, 5 + num_classes, num_grid * num_grid).contiguous()
    output = torch.squeeze(output, dim = 0)

    anchors_w = anchors[:, 0].contiguous().view(num_anchors, 1)
    anchors_h = anchors[:, 1].contiguous().view(num_anchors, 1)
    x_fit = torch.range(0, num_grid -1).repeat(num_grid)
    y_fit = torch.range(0, num_grid -1).repeat(num_grid).view(-1, num_grid).contiguous().transpose(0, 1).contiguous().view(1, -1)

    coord_box = torch.zeros_like(output[:, :4, :])
    coord_box[:, 0, :] = output[:, 0, :].sigmoid() + x_fit
    coord_box[:, 1, :] = output[:, 1, :].sigmoid() + y_fit
    coord_box[:, 2, :] = torch.exp(output[:, 2, :]) * anchors_w
    coord_box[:, 3, :] = torch.exp(output[:, 3, :]) * anchors_h
    conf_box = output[:, 4, :].sigmoid()
    cls_box = output[:, 5:, :].sigmoid()
    return coord_box, conf_box, cls_box

def test_image_processing(image, image_size):
    image = cv2.resize(image, (image_size, image_size))
    image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype = torch.float32)
    image = image.unsqueeze(0)
    return image

if __name__ == "__main__":
    help(iou)