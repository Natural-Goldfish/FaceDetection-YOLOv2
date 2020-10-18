from src.utils import iou
import torch
import copy
import math
from torch.nn.functional import sigmoid

class Custom_loss():
    def __init__(self, 
        batch_size = 10,
        num_classes = 1, 
        anchors = [(1.3221, 1.73145),
                    (3.19275, 4.00944),
                    (5.05587, 8.09892),
                    (9.47112, 4.84053),
                    (11.2364, 10.0071)], 
        coord_scale = 5,
        noobj_scale = 0.5,
        threshold = 0.6,
        reduction = 32):

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors, requires_grad = False)
        self.num_anchors = len(self.anchors)
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.num_grid = 13
        self.threshold = threshold
        self.reduction = reduction
        

    def __call__(self, output, label):
        pred_boxes, pcoord, pconf, pcls = self._pred_boxes(output)
        resp_coord_mask, tcoord, noobj_conf_mask, obj_conf_mask, tconf, resp_cls_mask, tcls= self._build_target(pred_boxes, label)

        resp_cls_mask = resp_cls_mask.view(-1, self.num_anchors, self.num_grid*self.num_grid, self.num_classes).contiguous().transpose(2, 3)
        tcls = tcls.view(-1, self.num_anchors, self.num_grid*self.num_grid, self.num_classes).contiguous().transpose(2, 3)

        pcls = pcls*resp_cls_mask.detach()
        tcls = tcls*resp_cls_mask.detach()

        mse = torch.nn.MSELoss(size_average= False)
        cel = torch.nn.CrossEntropyLoss()

        coord_loss = self.coord_scale * mse(pcoord * resp_coord_mask, tcoord * resp_coord_mask)/self.batch_size
        obj_conf_loss = mse(pconf * obj_conf_mask, tconf * obj_conf_mask)/self.batch_size
        noobj_conf_loss = self.noobj_scale * mse(pconf * noobj_conf_mask, tconf * noobj_conf_mask)/self.batch_size
        
        cls_loss = mse(pcls, tcls) / self.batch_size

        conf_loss = noobj_conf_loss + obj_conf_loss
        total_loss = coord_loss + conf_loss + cls_loss
        return total_loss, coord_loss, conf_loss, cls_loss

    def _pred_boxes(self, output):
        output = output.view(-1, self.num_anchors, 5 + self.num_classes, self.num_grid * self.num_grid)
        anchors_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchors_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        pcoord = torch.zeros_like(output[:, :, :4, :])
        pcoord[:, :, :2, :] = output[:, :, :2, :].sigmoid()
        pcoord[:, :, 2, :] = torch.exp(output[:, :, 2, :])
        pcoord[:, :, 2, :] = (pcoord[:, :, 2, :].detach() * anchors_w.detach().cuda()).sqrt()
        pcoord[:, :, 3, :] = torch.exp(output[:, :, 3, :])
        pcoord[:, :, 3, :] = (pcoord[:, :, 3, :].detach() * anchors_h.detach().cuda()).sqrt()

        pconf = output[:, :, 4, :].sigmoid()
        pcls = output[:, :, 5:, :].sigmoid()                        ## Ambiguous about adding sigmoid activation function

        pred_boxes = output[:, :, :4, :].clone().detach().requires_grad_(False)        
        
        if torch.cuda.is_available() :
            x_fit = torch.range(0, self.num_grid - 1).repeat(self.num_grid).cuda()
            y_fit = torch.range(0, self.num_grid - 1).repeat(self.num_grid).view(-1, self.num_grid).contiguous().transpose(0, 1).contiguous().view(1, -1).cuda()
            w_fit = torch.exp(pcoord[:, :, 2, :].detach()).cuda()
            h_fit = torch.exp(pcoord[:, :, 3, :].detach()).cuda()
        else:
            x_fit = torch.range(0, self.num_grid - 1).repeat(self.num_grid)
            y_fit = torch.range(0, self.num_grid - 1).repeat(self.num_grid).view(-1, self.num_grid).contiguous().transpose(0, 1).contiguous().view(1, -1)
            w_fit = torch.exp(pcoord[:, :, 2, :].detach())
            h_fit = torch.exp(pcoord[:, :, 3, :].detach())
        anchors_w.requires_grad_(False)
        anchors_h.requires_grad_(False)
        pred_boxes[:, :, 0, :] = pcoord[:, :, 0, :].detach() + x_fit
        pred_boxes[:, :, 1, :] = pcoord[:, :, 1, :].detach() + y_fit
        pred_boxes[:, :, 2, :] = w_fit * anchors_w.cuda()
        pred_boxes[:, :, 3, :] = h_fit * anchors_h.cuda()

        return pred_boxes, pcoord, pconf, pcls



    def _build_target(self, pred_boxes, label):
        # label = [xmin, ymin, width, height, class_id]
        batch_size = len(label)

        resp_coord_mask = torch.zeros_like(pred_boxes, requires_grad = False) 
        tcoord = torch.zeros_like(pred_boxes, requires_grad = False)

        noobj_conf_mask = torch.ones((batch_size, self.num_anchors, self.num_grid * self.num_grid), requires_grad = False).cuda()
        obj_conf_mask = torch.zeros((batch_size, self.num_anchors, self.num_grid * self.num_grid), requires_grad = False).cuda()
        tconf = torch.zeros((batch_size, self.num_anchors, self.num_grid * self.num_grid), requires_grad = False).cuda()

        resp_cls_mask = torch.zeros((batch_size, self.num_grid * self.num_grid * self.num_anchors, self.num_classes), requires_grad = False).byte().cuda()
        tcls = torch.zeros((batch_size, self.num_grid * self.num_grid * self.num_anchors, self.num_classes), requires_grad = False).cuda()

        anchor_bx = torch.zeros((len(self.anchors), 4)).cuda()
        anchor_bx[: , 2:4] = copy.deepcopy(self.anchors)
        anchor_bx[:, :2] = 0

        for batch_num in range(len(label)):
            if len(label[batch_num]) == 0 : continue

            ground_truth = torch.zeros((len(label[batch_num]), 5))

            for i, anno in enumerate(label[batch_num]):
                ground_truth[i, 0] = (anno[0] + anno[2]/2)/ self.reduction  # X_min -> X_c
                ground_truth[i, 1] = (anno[1] + anno[3]/2)/ self.reduction  # Y_min -> Y_c
                ground_truth[i, 2] = anno[2]/ self.reduction                # Width(Image)  -> Width(Feature map)
                ground_truth[i, 3] = anno[3]/ self.reduction                # Height(Image) -> He-ight(Feature map)
                ground_truth[i, 4] = torch.tensor(anno[4])

            ground_truth = ground_truth.cuda()

            cur_pred_box = pred_boxes[batch_num]
            cur_pred_box = cur_pred_box.transpose(1, 2).contiguous().view(-1, 4).contiguous()

            iou_gtpb = iou(ground_truth, cur_pred_box)
            temp_mask = (iou_gtpb > self.threshold).sum(0) >= 1
            temp_mask = (temp_mask.view(-1, self.num_grid * self.num_grid).contiguous()).sum(0) >= 1

            noobj_conf_mask[batch_num][temp_mask.expand_as(noobj_conf_mask[batch_num])] = 0

            ground_truth_wh = copy.deepcopy(ground_truth)
            ground_truth_wh[:, :2] = 0

            iou_gtab= iou(ground_truth_wh, anchor_bx)
            _, bestBb_gt = iou_gtab.max(1)

            for idx in range(len(ground_truth)):
                resp_bb_idx = bestBb_gt[idx]
                grid_x, grid_y = min(self.num_grid - 1, max(0, int(ground_truth[idx, 0]))), min(self.num_grid - 1, max(0, int(ground_truth[idx, 1])))
                i = self.num_grid * grid_y + grid_x
                resp_coord_mask[batch_num, resp_bb_idx , :, i] = 1
                tcoord[batch_num][resp_bb_idx][0][i] = ground_truth[idx, 0] - grid_x
                tcoord[batch_num][resp_bb_idx][1][i] = ground_truth[idx, 1] - grid_y
                tcoord[batch_num][resp_bb_idx][2][i] = ground_truth[idx, 2].sqrt()
                tcoord[batch_num][resp_bb_idx][3][i] = ground_truth[idx, 3].sqrt()

                obj_conf_mask[batch_num][resp_bb_idx][i] = 1
                tconf[batch_num][resp_bb_idx][i] = 1

                resp_cls_mask[batch_num][self.num_grid * self.num_grid * resp_bb_idx + i] = 1
                tcls[batch_num][self.num_grid * self.num_grid * resp_bb_idx + i] = ground_truth[idx][4]

        return resp_coord_mask, tcoord, noobj_conf_mask, obj_conf_mask, tconf, resp_cls_mask, tcls

if __name__ == "__main__" :
    Custom_loss()