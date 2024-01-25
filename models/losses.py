import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def weight_binary_cross_entropy_loss(input,target):
    # target = target.long
    n, c, _, _ = input.shape
    # mask = np.zeros(shape=(targets.shape[2],targets.shape[3])) #256,256
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    target = F.one_hot(target.long(), num_classes=2)
    target =target.permute(0, 3, 1, 2).float()

    # pos_weight = torch.tensor(1)
    criterion = torch.nn.BCEWithLogitsLoss() #包含sigmoid
    loss = criterion(input,target)

    return loss

def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    # bce = weight_binary_cross_entropy_loss(inputs,targets)

    n, c, _, _ = inputs.shape
    # mask = np.zeros(shape=(targets.shape[2],targets.shape[3])) #256,256
    if targets.dim() == 4:
        targets = torch.squeeze(targets, dim=1)
    targets = F.one_hot(targets.long(), num_classes=2)
    targets = targets.permute(0, 3, 1, 2).float()
    criterion = torch.nn.BCEWithLogitsLoss()  # 包含sigmoid
    bce = criterion(inputs, targets)

    smooth = 1e-5
    input = torch.sigmoid(inputs)
    num = targets.shape[0]
    input = input.reshape(num, -1)
    target = targets.reshape(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    labuda = 0.75
    loss = labuda*bce + (1-labuda)*dice
    # loss = bce + dice

    # inter = (inputs * targets).sum()
    # eps = 1e-5
    # dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return loss

#Boundary loss for RS images
def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    divce = label.device
    one_hot_label = torch.eye(
        n_classes, device='cuda', requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label

def BoundaryLoss(pred,gt,theta0=3, theta=3):
    "return boundary loss"
    n, c, _, _ = pred.shape

    # softmax so that predicted map can be distributed in [0, 1]
    pred = torch.softmax(pred, dim=1)
    if gt.dim() == 4:
        target = torch.squeeze(gt, dim=1)
    gt = F.one_hot(gt.long(), num_classes=2)
    gt =gt.permute(0, 3, 1, 2).float()
    # one-hot vector of ground truth
    # gt = gt.long()
    # gt = gt.squeeze(1)
    # one_hot_gt = one_hot(gt, c)

    # boundary map
    gt_b = F.max_pool2d(
        1 - gt, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    gt_b -= 1 - gt
    # gt_b_show = gt_b[0, 0, :, :]
    # gt_b_show = gt_b_show.data.cpu().numpy()
    # gt_b_show = np.asarray(gt_b_show, dtype=np.uint8)
    # gt_b_show = gt_b_show * 255
    # cv2.imwrite('E:/BL_test/bl/gt_b.png', gt_b_show)


    pred_b = F.max_pool2d(
        1 - pred, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    pred_b -= 1 - pred
    # pred_b_show = pred_b[0, 0, :, :]

    # pred_b_show = pred_b[0, :, :]
    # pred_b_show = pred_b_show.data.cpu().numpy()
    # pred_b_show = np.asarray(pred_b_show, dtype=np.uint8)
    # pred_b_show = pred_b_show * 255
    # cv2.imwrite('E:/BL_test/bl/pred_b.png', pred_b_show)

    # extended boundary map
    gt_b_ext = F.max_pool2d(
        gt_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)
    # gt_b_ext_show = gt_b_ext[0, 0, :, :]
    # gt_b_ext_show = gt_b_ext_show.data.cpu().numpy()
    # gt_b_ext_show = np.asarray(gt_b_ext_show, dtype=np.uint8)
    # gt_b_ext_show = gt_b_ext_show * 255
    # cv2.imwrite('E:/BL_test/bl/gt_b_ext.png', gt_b_ext_show)

    pred_b_ext = F.max_pool2d(
        pred_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)
    # pred_b_ext_show = pred_b_ext[0, 0, :, :]

    # pred_b_ext_show = pred_b_ext[0, :, :]
    # pred_b_ext_show = pred_b_ext_show.data.cpu().numpy()
    # pred_b_ext_show = np.asarray(pred_b_ext_show, dtype=np.uint8)
    # pred_b_ext_show = pred_b_ext_show * 255
    # cv2.imwrite('E:/BL_test/bl/pred_b_ext.png', pred_b_ext_show)


    # reshape
    # gt_b = gt_b.view(n, c, -1)
    # pred_b = pred_b.view(n, c, -1)
    # gt_b_ext = gt_b_ext.view(n, c, -1)
    # pred_b_ext = pred_b_ext.view(n, c, -1)
    gt_b = torch.flatten(gt_b,1)
    pred_b =torch.flatten(pred_b,1)
    gt_b_ext = torch.flatten(gt_b_ext,1)
    pred_b_ext = torch.flatten(pred_b_ext,1)

    # Precision, Recall
    P = torch.sum(pred_b * gt_b_ext) / (torch.sum(pred_b) + 1e-7)
    R = torch.sum(pred_b_ext * gt_b) / (torch.sum(gt_b) + 1e-7)

    # Boundary F1 Score
    BF1 = 2 * P * R / (P + R + 1e-7)

    # summing BF1 Score for each class and average over mini-batch
    loss = torch.mean(1 - BF1)

    return loss


class Boundary_ce_loss(nn.Module):
    def __init__(self, theta0=3, theta=3, weight=None, reduction='mean',ignore_index=255):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.weight=weight
        self.reduction=reduction
        self.ignore_index=ignore_index


    def forward(self, input, target,epoch_id,epoch_max):

        # target = target.long()
        # if target.dim() == 4:
        #     target = torch.squeeze(target, dim=1)
        # if input.shape[-1] != target.shape[-1]:
        #     input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
        # #loss_w = [0.1, 0.3, 0.5, 0.6, 0.7]
        # ce_loss=F.cross_entropy(input=input, target=target, weight=self.weight,
        #                            ignore_index=self.ignore_index, reduction=self.reduction)
        # print('ce_loss',ce_loss)
        # print('-----')
        # print('boundary_loss',boundary_loss)
        # print('-----')
        # i=epoch_id//40
        w0 = 0.05
        if epoch_id < int(epoch_max/2):
            w=0
            ce_loss = cross_entropy(input = input,target=target)
            loss = (1 - w) * ce_loss

        else:
            w = w0+0.01*(epoch_id-int(epoch_max/2))
            ce_loss = cross_entropy(input=input, target=target)
            boundary_loss = BoundaryLoss(pred=input, gt=target, theta0=3, theta=3)
            if w>0.5:
                w = 0.5
            loss = (1 - w) * ce_loss + w * boundary_loss


        # print('loss',loss)

        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        n, c, _, _ = input.shape
        # mask = np.zeros(shape=(targets.shape[2],targets.shape[3])) #256,256
        mask = target[0, 0, :, :]
        reversed_mask = mask
        reversed_mask = torch.where(reversed_mask == 1, 0, 1)
        i_m_reverse = reversed_mask
        mask_join = torch.zeros(size=(n, 2, mask.shape[0], mask.shape[1]))
        mask_join[:,0,...] = mask
        mask_join[:,1,...] = i_m_reverse
        mask_join = mask_join.to(device='cuda')

        num = mask_join.shape[0]
        input = input.reshape(num, -1)
        target = mask_join.reshape(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        '''

        # compute loss

        # logits = logits.float() # use fp32 if logits is fp16
        #logits是输入，label是标签
        # if label.dim() == 4:
        #     label = torch.squeeze(label, dim=1)
        # target = F.one_hot(label.long(), num_classes=2)
        # target = target.permute(0, 3, 1, 2).float()
        label = label.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)

        ce_loss = self.crit(logits, label)

        focal_loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        if self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss

class FocalLoss_with_dice(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss_with_dice, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):

        # compute loss
        # logits = logits.float() # use fp32 if logits is fp16

        n, c, _, _ = logits.shape
        # mask = label[0, 0, :, :]
        # reversed_mask = mask
        # reversed_mask = torch.where(reversed_mask == 1, 0, 1)
        # i_m_reverse = reversed_mask
        # mask_join = torch.zeros(size=(n, c, mask.shape[0], mask.shape[1]))
        # mask_join[:, 0, ...] = mask
        # mask_join[:, 1, ...] = i_m_reverse
        # label = mask_join.to(device='cuda')
        #
        # with torch.no_grad():
        #     alpha = torch.empty_like(logits).fill_(1 - self.alpha)
        #     alpha[label == 1] = self.alpha
        #
        # probs = torch.sigmoid(logits)
        # pt = torch.where(label == 1, probs, 1 - probs)

        if label.dim() == 4:
            label = torch.squeeze(label, dim=1)
        target = F.one_hot(label.long(), num_classes=2)
        target = target.permute(0, 3, 1, 2).float()

        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[target == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(target == 1, probs, 1 - probs)

        ce_loss = self.crit(logits, target)

        focal_loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        if self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        smooth=1e-5
        # logits=torch.sigmoid(logits)
        num=label.shape[0]
        probs=probs.reshape(num,-1)
        target=target.reshape(num,-1)
        # print(logits.type())
        # print(label.type())
        intersection=(probs*(target.float()))
        dice=(2.*intersection.sum(1)+smooth)/(probs.sum(1)+(target.float()).sum(1)+smooth)
        dice=1-dice.sum()/num
        loss = 0.5*focal_loss+0.5*dice
        return loss


class Focal_Dice_BL(nn.Module):

    def __init__(self):
        super(Focal_Dice_BL, self).__init__()

        self.Focal_loss = FocalLoss()
        self.Dice_loss = DiceLoss()

    def forward(self,input, target,epoch_id,epoch_max):
        if epoch_id < int(epoch_max/2):
            # focal = self.Focal_loss(inputs=input, targets=target, alpha=ALPHA, gamma=GAMMA)
            focal = self.Focal_loss(logits=input, label=target)
            dice = self.Dice_loss(input=input, target=target)
            loss = focal + 0.75*dice
        else:
            focal = self.Focal_loss(logits=input, label=target)
            dice = self.Dice_loss(input=input, target=target)
            # target = torch.squeeze(target, dim=1)
            boundary = BoundaryLoss(pred=input, gt=target, theta0=3, theta=3)
            w0=0.05
            w = w0+0.01*(epoch_id-int(epoch_max/2))
            if w>0.5:
                w = 0.5
            loss = (1-w)*(focal + 0.75*dice)+ w*boundary

        return loss

# if __name__ == '__main__':
#     pred = r'E:/BL_test/img2.png')
#     gt = cv2.imread(r'E:/BL_test/gt2.png')
#     label = np.array(Image.open(L_path), dtype=np.uint8)
#     bl_loss = Boundary_ce_loss(pred,gt)
#     print(bl_loss)