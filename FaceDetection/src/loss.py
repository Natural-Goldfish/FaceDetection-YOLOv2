from src.utils import iou
import torch
import copy
import math
from torch.nn.functional import sigmoid

_CUDA_FLAG = torch.cuda.is_available()

class Custom_loss():
    def __init__(self, anchors, batch_size, coord_scale, noobj_scale, num_classes = 1, 
        threshold = 0.6,
        reduction = 32):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors).view(-1, 2)
        self.num_anchors = len(self.anchors)
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.num_grid = 13
        self.threshold = threshold
        self.reduction = reduction

    def __call__(self, output, label):
        pred_boxes, pcoord, pconf, pcls = self._pred_boxes(output)
        resp_coord_mask, tcoord, noobj_conf_mask, obj_conf_mask, tconf, resp_cls_mask, tcls= self._build_target(pred_boxes, label)

        resp_cls_mask = resp_cls_mask.view(-1, self.num_classes).contiguous()
        pcls = pcls.transpose(2, 3).contiguous().view(-1, self.num_classes)
        pcls = resp_cls_mask * pcls
        tcls = tcls.view(-1)

        mse = torch.nn.MSELoss(size_average = False)
        bce = torch.nn.BCELoss(size_average = False)

        coord_loss = mse(self.coord_scale * pcoord * resp_coord_mask, self.coord_scale * tcoord * resp_coord_mask)/self.batch_size

        obj_conf_loss = mse(pconf * obj_conf_mask, tconf * obj_conf_mask)/self.batch_size
        noobj_conf_loss =  mse(self.noobj_scale * pconf * noobj_conf_mask, self.noobj_scale * tconf * noobj_conf_mask)/self.batch_size
        conf_loss = noobj_conf_loss + obj_conf_loss

        cls_loss = bce(pcls, tcls) / self.batch_size

        total_loss = coord_loss + conf_loss + cls_loss
        return total_loss, coord_loss, conf_loss, cls_loss

    def _pred_boxes(self, output):
        """
        This make prediction boxes for caculating loss. All of values's description which are used are below :
        - pcoord is for coordinate
        - pconf is for confidence
        - pcls is for class
        - x_fit and y_fit are added to calculate correct grid cell
        - w_fit and h_fit are used to calculate coorect width and height
        """
        self.batch_size = len(output)
        output = output.view(-1, self.num_anchors, 5 + self.num_classes, self.num_grid * self.num_grid)
        anchors_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchors_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        if _CUDA_FLAG : 
            anchors_w = anchors_w.cuda()
            anchors_h = anchors_h.cuda()

        pcoord = torch.zeros_like(output[:, :, :4, :])
        pcoord[:, :, :2, :] = output[:, :, :2, :].sigmoid()
        pcoord[:, :, 2, :] = (torch.exp(output[:, :, 2, :]) * anchors_w).sqrt()
        pcoord[:, :, 3, :] = (torch.exp(output[:, :, 3, :]) * anchors_h).sqrt()
        pconf = output[:, :, 4, :].sigmoid()
        pcls = output[:, :, 5:, :].sigmoid()
        pred_boxes = output[:, :, :4, :].clone().detach().requires_grad_(False)
        
        if _CUDA_FLAG :       
            x_fit = torch.range(0, self.num_grid - 1).repeat(self.num_grid).cuda()
            y_fit = torch.range(0, self.num_grid - 1).repeat(self.num_grid).view(-1, self.num_grid).contiguous().transpose(0, 1).contiguous().view(1, -1).cuda()
        else:
            x_fit = torch.range(0, self.num_grid - 1).repeat(self.num_grid)
            y_fit = torch.range(0, self.num_grid - 1).repeat(self.num_grid).view(-1, self.num_grid).contiguous().transpose(0, 1).contiguous().view(1, -1)

        pred_boxes[:, :, 0, :] = pcoord[:, :, 0, :].detach() + x_fit
        pred_boxes[:, :, 1, :] = pcoord[:, :, 1, :].detach() + y_fit
        pred_boxes[:, :, 2, :] = pcoord[:, :, 2, :].detach() * pcoord[:, :, 2, :].detach()
        pred_boxes[:, :, 3, :] = pcoord[:, :, 3, :].detach() * pcoord[:, :, 3, :].detach()
        return pred_boxes, pcoord, pconf, pcls

    def _build_target(self, pred_boxes, label):
        """
        Define responsible masks and targets
        """

        # Coordinate
        resp_coord_mask = torch.zeros_like(pred_boxes, requires_grad = False) 
        tcoord = torch.zeros_like(pred_boxes, requires_grad = False)

        # Confidence
        noobj_conf_mask = torch.ones((self.batch_size, self.num_anchors, self.num_grid * self.num_grid), requires_grad = False)
        obj_conf_mask = torch.zeros((self.batch_size, self.num_anchors, self.num_grid * self.num_grid), requires_grad = False)
        tconf = torch.zeros((self.batch_size, self.num_anchors, self.num_grid * self.num_grid), requires_grad = False)

        # Class
        resp_cls_mask = torch.zeros((self.batch_size, self.num_grid * self.num_grid * self.num_anchors, self.num_classes), requires_grad = False)
        tcls = torch.zeros((self.batch_size, self.num_grid * self.num_grid * self.num_anchors, 1), requires_grad = False)

        # Anchor boxes
        anchor_bx = torch.zeros((len(self.anchors), 4))
        anchor_bx[: , 2:4] = copy.deepcopy(self.anchors)
        anchor_bx[:, :2] = 0

        if _CUDA_FLAG :
            resp_coord_mask = resp_coord_mask.cuda()
            tcoord = tcoord.cuda()
            noobj_conf_mask = noobj_conf_mask.cuda()
            obj_conf_mask = obj_conf_mask.cuda()
            tconf = tconf.cuda()
            resp_cls_mask = resp_cls_mask.cuda()
            tcls = tcls.cuda()
            anchor_bx = anchor_bx.cuda()

        for cur_batch in range(self.batch_size):
            # If there isn't any object, go next
            if len(label[cur_batch]) == 0 : continue

            # Make ground-truth for current batch
            ground_truth = torch.zeros((len(label[cur_batch]), 5))
            for i, anno in enumerate(label[cur_batch]):
                ground_truth[i, 0] = (anno[0] + anno[2]/2)/ self.reduction  # X_min -> X_c
                ground_truth[i, 1] = (anno[1] + anno[3]/2)/ self.reduction  # Y_min -> Y_c
                ground_truth[i, 2] = anno[2]/ self.reduction                # Width(Image)  -> Width(Feature map)
                ground_truth[i, 3] = anno[3]/ self.reduction                # Height(Image) -> Height(Feature map)
                ground_truth[i, 4] = anno[4].byte()                         # class index
            if _CUDA_FLAG :
                ground_truth = ground_truth.cuda()

            cur_pred_box = pred_boxes[cur_batch]
            cur_pred_box = cur_pred_box.transpose(1, 2).contiguous().view(-1, 4).contiguous()

            # In order to find which boxes are not responsible
            iou_gtpb = iou(ground_truth, cur_pred_box)
            obj_exist_mask = (iou_gtpb > self.threshold).sum(0) >= 1
            obj_exist_mask = obj_exist_mask.view(-1, self.num_grid * self.num_grid)
            noobj_conf_mask[cur_batch][obj_exist_mask] = 0

            # In order to find best anchor box for objects
            ground_truth_wh = copy.deepcopy(ground_truth)
            ground_truth_wh[:, :2] = 0
            iou_gtab = iou(ground_truth_wh, anchor_bx)
            _, best_bbx_idx = iou_gtab.max(1)

            for idx in range(len(ground_truth)):
                resp_bbx_idx = best_bbx_idx[idx]
                grid_x, grid_y = min(self.num_grid - 1, max(0, int(ground_truth[idx, 0]))), min(self.num_grid - 1, max(0, int(ground_truth[idx, 1])))
                i = self.num_grid * grid_y + grid_x
                resp_coord_mask[cur_batch, resp_bbx_idx, :, i] = 1
                tcoord[cur_batch][resp_bbx_idx][0][i] = ground_truth[idx, 0] - grid_x
                tcoord[cur_batch][resp_bbx_idx][1][i] = ground_truth[idx, 1] - grid_y
                tcoord[cur_batch][resp_bbx_idx][2][i] = ground_truth[idx, 2].sqrt()
                tcoord[cur_batch][resp_bbx_idx][3][i] = ground_truth[idx, 3].sqrt()

                obj_conf_mask[cur_batch][resp_bbx_idx][i] = 1
                tconf[cur_batch][resp_bbx_idx][i] = 1

                resp_cls_mask[cur_batch][self.num_grid * self.num_grid * resp_bbx_idx + i] = 1
                tcls[cur_batch][self.num_grid * self.num_grid * resp_bbx_idx + i] = ground_truth[idx][4]
        return resp_coord_mask, tcoord, noobj_conf_mask, obj_conf_mask, tconf, resp_cls_mask, tcls
