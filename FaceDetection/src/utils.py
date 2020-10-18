import torch
import cv2
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
    #Union = (area1 + area2.t()) - InterSection
    Union = (area1 + area2) - InterSection


    return InterSection / Union

def pred_boxes(output, num_anchors = 5, num_classes = 1, num_grid = 13, anchors= []):
    output = output.view(num_anchors, 5 + num_classes, num_grid * num_grid).contiguous()

    x_fit = torch.range(0, num_grid -1).repeat(num_grid)
    y_fit = torch.range(0, num_grid -1).repeat(num_grid).view(-1, num_grid).contiguous().transpose(0, 1).contiguous().view(1, -1)

    anchors = torch.tensor(anchors).clone().detach()
    anchors_w = anchors[:, 0].contiguous().view(-1, 1)
    anchors_h = anchors[:, 1].contiguous().view(-1, 1)

    pred_box = torch.tensor(output).clone().detach().cuda()
    pred_box[:, 0, :] = output[:, 0, :].sigmoid() + x_fit
    pred_box[:, 1, :] = output[:, 1, :].sigmoid() + y_fit
    pred_box[:, 2, :] = torch.exp(output[:, 2, :]) * anchors_w
    pred_box[:, 3, :] = torch.exp(output[:, 3, :]) * anchors_h
    pred_box[:, 4, :] = output[:, 4, :].sigmoid()
    pred_box[:, 5:, :] = output[:, 5:, :].sigmoid()

    return pred_box

def draw_rec(img, min_x, min_y, max_x, max_y):

    for i in range(len(min_x)):
        img = cv2.rectangle(img, (min_x[i], min_y[i]), (max_x[i], max_y[i]), (255, 0, 0), 3)

    cv2.imshow("sdf", img)
    cv2.waitKey(0)

def nms(pred_box):
    pred_box = pred_box.transpose(1, 2).contiguous().view(13*13*5, 6).contiguous().t()
    mask = pred_box[4, :] > 0.8
    target = pred_box[:, mask]
    min_x = target[0, :]*32 - (target[2, :]/2)*32
    min_y = target[1, :]*32 - (target[3, :]/2)*32
    max_x = target[0, :]*32 + (target[2, :]/2)*32
    max_y = target[1, :]*32 + (target[3, :]/2)*32
    print(pred_box[4,:].max())
    return min_x, min_y, max_x, max_y

if __name__ == "__main__":
    help(iou)