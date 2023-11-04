import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models as tvmodels
import torchvision
from efficientnet_pytorch import EfficientNet
import math
import numpy as np
from torchvision import transforms


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.sobel_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        sobel_kernel = torch.tensor([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=torch.float32)
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        self.sobel_conv.weight.data = sobel_kernel
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # transforms.ToTensor()
        ])
    
    def forward(self, x):
        if x.shape[1] !=1:
            x = self.transform(x)
        x = torch.sigmoid(x)
        edge = self.sobel_conv(x)
        return edge


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConstantNormalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ConstantNormalize, self).__init__()
        mean = torch.Tensor(mean).view([1, 3, 1, 1])
        std = torch.Tensor(std).view([1, 3, 1, 1])
        # https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def forward(self, x):
        return (x  - self.mean) / (self.std + 1e-5)


class Conv1x1(nn.Sequential):
     def __init__(self, in_planes, out_planes=16, has_se=False, se_reduction=None):
        if has_se:
            if se_reduction is None:
                # se_reduction= int(math.sqrt(in_planes))
                se_reduction = 2
            super(Conv1x1, self).__init__(SELayer(in_planes, se_reduction),
                                           nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                           nn.BatchNorm2d(out_planes),
                                           nn.LeakyReLU()
                                           )
        else:
            super(Conv1x1, self).__init__(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                          nn.BatchNorm2d(out_planes),
                                          nn.LeakyReLU()
                                         )

# https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnext50_32x4d
class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.in_planes = in_planes
        self.out_planes = out_planes
        if self.in_planes != self.out_planes:
            self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x
        if self.in_planes != self.out_planes:
            identity = self.conv3(identity)
            identity = self.bn3(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class FRUnit(nn.Module):
    """
    Factorisation and Reweighting unit
    """
    def __init__(self, channels=32, mu_init=0.5, reweight_mode='manual', normalize=True):
        super(FRUnit, self).__init__()
        assert reweight_mode in ['manual', 'constant', 'learnable', 'nn']
        self.mu = mu_init
        self.reweight_mode = reweight_mode
        self.normalize = normalize
        self.inv_conv = ResBlock(channels, channels)
        self.var_conv = ResBlock(channels, channels)
        # self.mix_conv = ResBlock(2*channels, channels)
        # self.mix_conv = Conv1x1(2*channels, channels)
        if reweight_mode == 'learnable':
            self.mu = nn.Parameter(torch.tensor(mu_init))
            # self.x = nn.Parameter(torch.tensor(0.))
            # self.mu = torch.sigmoid(self.x)
        elif reweight_mode == 'nn':
            self.fc = nn.Sequential(nn.Linear(channels, 1),
                                    nn.Sigmoid()
                                   )
        else:
            self.mu = mu_init


    def forward(self, feat):
        shad_feat = self.inv_conv(feat)
        noshad_feat = self.var_conv(feat)
        # noshad_feat = feat - shad_feat

        if self.normalize:
            shad_feat = F.normalize(shad_feat)
            noshad_feat = F.normalize(noshad_feat)
            # noshad_feat = shad_feat - (shad_feat * noshad_feat).sum(keepdim=True, dim=1) * shad_feat
            # noshad_feat = F.normalize(noshad_feat)
        
        if self.reweight_mode == 'nn':
            gap = feat.mean([2, 3])
            self.mu = self.fc(gap).view(-1, 1, 1, 1)

        # mix_feat = self.mu * noshad_feat + (1 - self.mu) * shad_feat
        mix_feat = noshad_feat + shad_feat
        # mix_feat = self.mix_conv(torch.cat([shad_feat, noshad_feat], dim=1))
        # print(self.mu)
        return shad_feat, noshad_feat, mix_feat


    def set_mu(self, val):
        assert self.reweight_mode == 'manual'
        self.mu = val


ml_features = []

def feature_hook(module, fea_in, fea_out):
#     print("hooker working")
    # module_name.append(module.__class__)
    # features_in_hook.append(fea_in)
    global ml_features
    ml_features.append(fea_out)
    return None


class ShadFilter(nn.Module):
    def __init__(self, in_channels):
        super(ShadFilter, self).__init__()
        # self.layer1 = nn.Linear(in_channels, in_channels)
        self.layer1 = nn.Linear(in_channels**2, in_channels**2)
        # self.layer1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=False)
        # self.layer1 = nn.Conv2d(1, in_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.relu = nn.LeakyReLU()
        self.layer2 = nn.Linear(in_channels**2, in_channels**2)                                                                                                                                                                                                                                                                                                               
        # self.layer2 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1, bias=False)
        # self.layer2 = nn.Conv2d(in_channels//2, in_channels//2, stride=1, kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(1)
        # self.bn3 = nn.BatchNorm2d(in_channels//2)

    def gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)   #C和w*h转置
        gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
        return gram

    def forward(self, x):
        # x = torch.triu(self.gram_matrix(x), diagonal=0)
        x = self.gram_matrix(x)
        (b, h, w) = x.size()
        # x = x.view(b, 1, h, w)
        x = x.view(b, h*w)
        x = self.layer1(x)
        # x = self.bn1(x)
        x = F.normalize(x)
        x = self.relu(x)
        x = self.layer2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = F.normalize(x)
        # x = self.bn3(x)
        # print(x[0])

        # print(x.size())

        return x
    

class ShadFilter2(nn.Module):
    def __init__(self, inp_size):
        super(ShadFilter2, self).__init__()
        self.layer1 = nn.Linear(inp_size, inp_size//2)
        self.layer2 = nn.Linear(inp_size//2, inp_size//4)
        self.layer3 = nn.Linear(inp_size//4, 64)
        self.relu = nn.LeakyReLU()

    def gram_matrix(self, y):
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        return gram
    
    def forward(self, x):
        gm = self.gram_matrix(x)
        b, _, _ = gm.size()
        tgm = gm[torch.triu(torch.ones(gm.size()[0], gm.size()[1], gm.size()[2]))==1].view(b, -1)
        out = self.layer1(tgm)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out
    

class LEModule(nn.Module):
    def __init__(self, lf_ch, hf_ch, out_ch):
        super(LEModule, self).__init__()
        self.conv0 = nn.Conv2d(hf_ch, hf_ch, kernel_size=1, padding=0)
        self.batch0 = nn.BatchNorm2d(hf_ch)
        self.relu0 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(hf_ch, hf_ch, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(hf_ch)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(hf_ch, hf_ch, kernel_size=5, padding=2)
        self.batch2 = nn.BatchNorm2d(hf_ch)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = Conv1x1(3*hf_ch, hf_ch)
        self.conv4 = Conv1x1(lf_ch+hf_ch, out_ch)
    
    def forward(self, low_feat, high_feat):
        x0 = self.conv0(low_feat)
        x0 = self.batch0(x0)
        x0 = self.relu0(x0)
        x1 = self.conv1(low_feat)
        x1 = self.batch1(x1)
        x1 = self.relu1(x1)
        x2 = self.conv2(low_feat)
        x2 = self.batch2(x2)
        x2 = self.relu2(x2)
        x3 = self.conv3(torch.cat([x0,x1,x2], dim=1))
        x4 = self.conv4(torch.cat([high_feat,x3], dim=1))
        return x4


class FDUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FDUnit, self).__init__()
        self.shad_conv = ResBlock(in_ch, out_ch)
        self.shad_bn = nn.BatchNorm2d(out_ch)
        self.shad_fc = nn.Sequential(nn.Linear(out_ch, 1),
                                     nn.Sigmoid())
        self.noshad_conv = ResBlock(in_ch, out_ch)
        self.noshad_bn = nn.BatchNorm2d(out_ch)
        self.noshad_fc = nn.Sequential(nn.Linear(out_ch, 1),
                                     nn.Sigmoid())

    def forward(self, x):
        shad_feat = self.shad_conv(x)
        shad_feat = self.shad_bn(shad_feat)
        alpha = shad_feat.mean([2,3])
        alpha = self.shad_fc(alpha).view(-1, 1, 1, 1)
        noshad_feat = self.noshad_conv(x)
        noshad_feat = self.noshad_bn(noshad_feat)
        beta = noshad_feat.mean([2,3])
        beta = self.noshad_fc(beta).view(-1, 1, 1, 1)
        mix_feat = alpha * shad_feat + beta * noshad_feat
        return shad_feat, noshad_feat, mix_feat


class SDDNet(nn.Module):
    # decompose net
    def __init__(self,
                 backbone='efficientnet-b0',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 dropout_2d=0,
                 normalize=False,
                 mu_init=0.5,
                 reweight_mode='constant'):

        super(SDDNet, self).__init__()

        self.mu_init = mu_init
        # self.reweight_mode = reweight_mode
        self.reweight_mode = 'nn'

        # load backbone
        if use_pretrained:
            self.feat_net = EfficientNet.from_pretrained(backbone)
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        else:
            self.feat_net = EfficientNet.from_name(backbone)

        # load vgg
        # vgg = torchvision.models.vgg16(pretrained=True)

        # remove classification head to get correct param count
        self.feat_net._avg_pooling=None
        self.feat_net._dropout=None
        self.feat_net._fc=None

        # register hook to extract multi-level features
        in_planes = []
        feat_layer_ids = list(range(0, len(self.feat_net._blocks), 2))
        for idx in feat_layer_ids:
            self.feat_net._blocks[idx].register_forward_hook(hook=feature_hook)
            in_planes.append(self.feat_net._blocks[idx]._bn2.num_features)

        if fix_backbone:
            for param in self.feat_net.parameters():
                param.requires_grad = False

        self.norm = ConstantNormalize()

        # 1*1 projection conv
        proj_convs = [ Conv1x1(ip, proj_planes, has_se=has_se) for ip in in_planes ]
        self.proj_convs = nn.ModuleList(proj_convs)

        # two stream feature
        # self.stem_conv = Conv1x1(proj_planes*len(in_planes), pred_planes, has_se=has_se)
        self.lower_conv = Conv1x1(proj_planes*3, pred_planes, has_se=has_se)
        self.higher_conv = Conv1x1(proj_planes*10, pred_planes, has_se=has_se)
        # self.lower_conv = ResBlock(proj_planes*3, pred_planes)
        # self.higher_conv = ResBlock(proj_planes*10, pred_planes)
        self.fr_lower = FRUnit(pred_planes, mu_init=self.mu_init, reweight_mode=self.reweight_mode, normalize=normalize)
        self.fr_higher = FRUnit(pred_planes, mu_init=self.mu_init, reweight_mode=self.reweight_mode, normalize=normalize)
        # self.fr_lower = FDUnit(pred_planes, pred_planes)
        # self.fr_higher = FDUnit(pred_planes, pred_planes)
        # self.fr_lower = FRUnit(pred_planes, mu_init=0.7, reweight_mode=self.reweight_mode, normalize=normalize)
        # self.fr_higher = FRUnit(pred_planes, mu_init=0.6, reweight_mode=self.reweight_mode, normalize=normalize)
        self.mix_conv = Conv1x1(pred_planes*2, pred_planes, has_se=True)
        self.noshad_conv = Conv1x1(pred_planes*2, pred_planes, has_se=True)
        self.shad_conv = Conv1x1(pred_planes*2, pred_planes, has_se=True)
        # self.mix_conv = LEModule(pred_planes, pred_planes, pred_planes)
        # self.noshad_conv = LEModule(pred_planes, pred_planes, pred_planes)
        # self.shad_conv = LEModule(pred_planes, pred_planes, pred_planes)
        # self.fc = nn.Linear(pred_planes, 1)
  
        # prediction   
        pred_layers_shadimg = []
        pred_layers_shadmask = []
        pred_layers_maskimg = []
        pred_layers_noshad = []
        if dropout_2d > 1e-6:
            pred_layers_shadimg.append(nn.Dropout2d(p=dropout_2d))
            pred_layers_shadmask.append(nn.Dropout2d(p=dropout_2d))
            pred_layers_maskimg.append(nn.Dropout2d(p=dropout_2d))
            pred_layers_noshad.append(nn.Dropout2d(p=dropout_2d))
        pred_layers_shadimg.append(nn.Conv2d(pred_planes, 3, 1))
        # pred_layers_shadimg.append(nn.BatchNorm2d(3))
        pred_layers_shadmask.append(nn.Conv2d(pred_planes, 1, 1))
        # pred_layers_shadmask.append(nn.BatchNorm2d(1))
        pred_layers_maskimg.append(nn.Conv2d(pred_planes, 3, 1))
        # pred_layers_maskimg.append(nn.BatchNorm2d(3))
        pred_layers_noshad.append(nn.Conv2d(pred_planes, 3, 1))
        # pred_layers_noshad.append(nn.BatchNorm2d(3))
        self.pred_conv_shadimg = nn.Sequential(*pred_layers_shadimg)
        self.pred_conv_shadmask = nn.Sequential(*pred_layers_shadmask)
        self.pred_conv_maskimg = nn.Sequential(*pred_layers_maskimg)
        self.pred_conv_noshad = nn.Sequential(*pred_layers_noshad)

        # self.conv1x1 = nn.Conv2d(pred_planes*2, 3, 1)
        self.pre_conv1x1 = nn.Conv2d(4, 3, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        # self.sobel_conv = Sobel()
        self.shad_filter_low= ShadFilter(pred_planes)
        self.shad_filter_high = ShadFilter(pred_planes)
        # self.shad_filter_low= ShadFilter2(528)
        # self.shad_filter_high = ShadFilter2(528)
        
    def forward(self, x):
        global ml_features
        
        b, c, h, w = x.size()
        ml_features = []

        if c == 4:
            x = self.pre_conv1x1(x)

        _ = self.feat_net.extract_features(self.norm(x))
        
        h_f, w_f = ml_features[0].size()[2:]
        # h_f, w_f = ml_features[2].size()[2:]
        # print(h_f, w_f)
        proj_features = []
        
        for i in range(3):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature_1 = torch.cat(proj_features, dim=1)
        stem_feat_low = self.lower_conv(cat_feature_1)


        proj_features.clear()
        for i in range(3, len(ml_features)):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature_2 = torch.cat(proj_features, dim=1)
        stem_feat_high = self.higher_conv(cat_feature_2)

        # stem_feat = self.stem_conv(cat_feature)
        
        # factorised feature
        # shad_feat, noshad_feat, mix_feat = self.fr(stem_feat)
        low_shad, low_noshad, low_feat = self.fr_lower(stem_feat_low)
        high_shad, high_noshad, high_feat = self.fr_higher(stem_feat_high)
        shad_feat = self.shad_conv(torch.cat([low_shad, high_shad], dim=1))
        noshad_feat = self.noshad_conv(torch.cat([low_noshad, high_noshad], dim=1))
        mix_feat = self.mix_conv(torch.cat([low_feat, high_feat], dim=1))
        # shad_feat = self.shad_conv(low_shad, high_shad)
        # noshad_feat = self.noshad_conv(low_noshad, high_noshad)
        # mix_feat = self.mix_conv(low_feat, high_feat)

        f_low_shad = self.shad_filter_low(low_shad)
        f_low_noshad = self.shad_filter_low(low_noshad)
        f_low_feat = self.shad_filter_low(low_feat)
        f_high_shad = self.shad_filter_high(high_shad)
        f_high_noshad = self.shad_filter_high(high_noshad)
        f_high_feat = self.shad_filter_high(high_feat)
        
        if self.training:
            # logits = F.interpolate(self.pred_conv(mix_feat), size=(h, w), mode='bilinear')
            # g = self.fc(var_feat.mean([2, 3]))
            # return logits, inv_feat, g
            # mix_feat = F.interpolate(mix_feat, size=(h, w), mode='bilinear')
            # logits_shadimg = self.pred_conv_shadimg(mix_feat)
            logits_shadimg = F.interpolate(self.pred_conv_shadimg(mix_feat), size=(h, w), mode='bilinear')
            # train_mask = self.pred_conv_shadmask(shad_feat)
            # shad_feat = F.interpolate(shad_feat, size=(h, w), mode='bilinear')
            # logits_shadmask = self.pred_conv_shadmask(shad_feat)
            logits_shadmask = F.interpolate(self.pred_conv_shadmask(shad_feat), size=(h, w), mode='bilinear')
            # train_mask = (F.sigmoid(train_mask) > 0.4).type(torch.int64)
            # f_noshad_region = train_mask * noshad_feat
            # f_mask_region = train_mask * shad_feat
            # logits_maskimg = self.pred_conv_shadimg(shad_feat)
            logits_maskimg = F.interpolate(self.pred_conv_shadimg(shad_feat), size=(h, w), mode='bilinear')
            # logits_maskimg = F.interpolate(self.pred_conv_shadimg(shad_feat), size=(h, w), mode='bilinear')
            # noshad_feat = F.interpolate(noshad_feat, size=(h, w), mode='bilinear')
            # logits_noshad = self.pred_conv_noshad(noshad_feat)
            logits_noshad = F.interpolate(self.pred_conv_noshad(noshad_feat), size=(h, w), mode='bilinear')
            # logits_noshad = F.interpolate(self.pred_conv_shadimg(noshad_feat), size=(h, w), mode='bilinear')
            # selfsup_noshad = F.interpolate(self.conv1x1(torch.cat([shad_feat, mix_feat], dim=1)), size=(h, w), mode='bilinear')

            # logits_shadimg = F.sigmoid(logits_shadimg)
            # logits_shadmask = F.sigmoid(logits_shadmask)
            # logits_noshad = F.sigmoid(logits_noshad)

            # sobel_shadmask = self.sobel_conv(logits_shadmask)
            # sobel_noshad = self.sobel_conv(logits_noshad)
            # sobel_noshad = self.sobel_conv(logits_shadimg)
            # print(logits_shadmask)
            return logits_shadimg, logits_shadmask, logits_noshad, \
                f_low_shad, f_high_shad, f_low_noshad, f_high_noshad, f_low_feat, f_high_feat, mix_feat,shad_feat,noshad_feat
        else:
            # if self.reweight_mode != 'learnable': 
            #     mix_feat = self.mu_init * noshad_feat + (1 - self.mu_init) * shad_feat
            #     # logits = F.interpolate(self.pred_conv_shad(mix_feat), size=(h, w), mode='bilinear')
            #     logits_shadmask = F.interpolate(self.pred_conv_shadmask(shad_feat), size=(h, w), mode='bilinear')
            #     # logits_maskimg = F.interpolate(self.pred_conv_maskimg(shad_feat), size=(h, w), mode='bilinear')
            #     logits_maskimg = F.interpolate(self.pred_conv_shadimg(shad_feat), size=(h, w), mode='bilinear')
            #     # logits_noshad = F.interpolate(self.pred_conv_noshad(noshad_feat), size=(h, w), mode='bilinear')
            #     logits_noshad = F.interpolate(self.pred_conv_shadimg(noshad_feat), size=(h, w), mode='bilinear')
            #     # selfsup_noshad = F.interpolate(self.conv1x1(torch.cat([shad_feat, mix_feat], dim=1)), size=(h, w), mode='bilinear')
            #     logits_shadimg = F.interpolate(self.pred_conv_shadimg(mix_feat), size=(h, w), mode='bilinear')
            # else:

            # seems no need for if, i can strictly use this else branch

            # logits = F.interpolate(self.pred_conv(mix_feat), size=(h, w), mode='bilinear')
            logits_shadmask = F.interpolate(self.pred_conv_shadmask(shad_feat), size=(h, w), mode='bilinear')
            # logits_maskimg = F.interpolate(self.pred_conv_maskimg(shad_feat), size=(h, w), mode='bilinear')
#            logits_maskimg = F.interpolate(self.pred_conv_shadimg(shad_feat), size=(h, w), mode='bilinear')
#            logits_noshad = F.interpolate(self.pred_conv_noshad(noshad_feat), size=(h, w), mode='bilinear')
            # logits_noshad = F.interpolate(self.pred_conv_shadimg(noshad_feat), size=(h, w), mode='bilinear')
            # selfsup_noshad = F.interpolate(self.conv1x1(torch.cat([shad_feat, mix_feat], dim=1)), size=(h, w), mode='bilinear')
#            logits_shadimg = F.interpolate(self.pred_conv_shadimg(mix_feat), size=(h, w), mode='bilinear')
            
            # sobel_shadmask = self.sobel_conv(logits_shadmask)
            # sobel_noshad = self.sobel_conv(logits_noshad)

            # logits_shadmask = torch.sigmoid(logits_shadmask)
            # logits_shadimg = torch.sigmoid(logits_shadimg)
            # logits_noshad = torch.sigmoid(logits_noshad)
            # logits_maskimg = torch.sigmoid(logits_maskimg)

            # sobel_noshad = self.sobel_conv(logits_shadimg)
#            return {'logit': logits_shadmask, 'noshad': logits_noshad, 'shad':logits_shadimg, 'maskimg': logits_maskimg}
            return {'logit': logits_shadmask}
            # return {'logit': logits_shadmask, 'noshad': logits_noshad, 'supervised': selfsup_noshad, 'shad':logits_shadimg}


