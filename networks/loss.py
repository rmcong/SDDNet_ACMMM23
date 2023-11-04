import torch
from torch import nn
import torch.nn.functional as F


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

    
class DiceLoss(nn.Module):
    def __init__(self, apply_sigmoid=True, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, pred, target):
        if self.apply_sigmoid:
            pred = F.sigmoid(pred)
            
        numerator = 2 * torch.sum(pred * target) + self.smooth
        denominator = torch.sum(pred + target) + self.smooth
        return 1 - numerator / denominator


class EdgeLoss(nn.Module):
    def __init__(self, apply_sigmoid=True):
        super(EdgeLoss, self).__init__()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, edge_shadmask, edge_noshad):
        if self.apply_sigmoid:
            edge_shadmask = F.sigmoid(edge_shadmask)
            edge_noshad = F.sigmoid(edge_noshad)
        # can't for backward gradient computation
        # edge_shadmask[edge_shadmask >= 0.5] = 1
        # edge_shadmask[edge_shadmask < 0.5] =0
        # edge_noshad[edge_noshad >= 0.5] = 1
        # edge_noshad[edge_noshad < 0.5] = 0
        edge = edge_shadmask + edge_noshad
        # print(edge)
        edge[edge<1] = 0
        edge[edge>=1] = 1
        numerator = torch.sum(edge)
        denominator = torch.sum(1. - edge)
        loss = numerator / denominator
        return loss

class OrthoLoss(nn.Module):
    def __init__(self):
        super(OrthoLoss, self).__init__()
        
    def forward(self, pred, target):
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)
        
        pred_ = pred
        target_ = target
        ortho_loss = 0
        dim = pred.shape[1]
        for i in range(pred.shape[0]):
            # ortho_loss += torch.mean(torch.abs(pred_[i:i+1,:].mm(target_[i:i+1,:].t()))/dim)
            ortho_loss += torch.mean((pred_[i:i+1,:].mm(target_[i:i+1,:].t())).pow(2)/dim)
        
        ortho_loss /= pred.shape[0]
        return ortho_loss

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ortho = OrthoLoss()
    
    def forward(self, img, noshad, mask):
        # batch_size = noshad.size(0)
        # mask = (mask > 0.5).type(torch.int64)
        img_noshad = img * (1-mask)
        noshad_noshad = noshad * (1-mask)
        img_shad = img * mask
        noshad_shad = noshad * mask

        loss = self.l1(img_noshad, noshad_noshad)
        # loss += 0.01*self.ortho(img_shad, noshad_shad)
        return loss

class DiffLoss_2(nn.Module):
    def __init__(self):
        super(DiffLoss_2, self).__init__()
        self.l1 = nn.L1Loss()
        self.ortho = OrthoLoss()
    
    def forward(self, img, noshad, label, mask):
        # batch_size = noshad.size(0)
        mask = (mask > 0.5).type(torch.int64)
        # mask = torch.sigmoid(mask)
        img_noshad = img * (1-mask)
        noshad_noshad = noshad * (1-label)

        loss = self.l1(img_noshad, noshad_noshad)
        # loss += 0.01*self.ortho(img_shad, noshad_shad)
        return loss

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.l1 = nn.L1Loss()
        
    # def forward(self, img, noshad, mask):
    #     # batch_size = noshad.size(0)
    #     hmask = (mask > 0.5).type(torch.int64)
    #     smask = (mask > 0.25).type(torch.int64)
    #     img_noshad = img * (smask-hmask)
    #     noshad_shad = noshad * hmask

    #     count_noshad = torch.sum(1.-hmask)

    #     lns = torch.mean(img_noshad)
    #     ls = torch.mean(noshad_shad)

    #     loss = torch.abs(lns-ls)*count_noshad
    #     return loss

    def gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)   #C和w*h转置
        gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
        return gram

    def forward(self, img, noshad):
        # mask = (mask > 0.5).type(torch.int64)
        # img_noshad = img * (1-mask)
        # # noshad_shad = noshad * mask
        # noshad_shad = noshad

        # g_noshad = self.gram_matrix(img_noshad)
        # g_shad = self.gram_matrix(noshad_shad)
        g_noshad = self.gram_matrix(img)
        g_shad = self.gram_matrix(noshad)

        loss = 0
        for i in range(g_shad.shape[0]):
            loss += torch.mean(torch.abs(g_noshad[i:i+1,:] - g_shad[i:i+1,:]))
        return loss

class ZeroLoss(nn.Module):
    def __init__(self):
        super(ZeroLoss, self).__init__()
        
    def forward(self, target):
        zero_loss = torch.mean(torch.abs(target))
        return zero_loss