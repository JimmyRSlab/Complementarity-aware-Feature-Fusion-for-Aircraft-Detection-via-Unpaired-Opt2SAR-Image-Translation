import torch
import torch.nn as nn
import warnings
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector


class GWF(nn.Module):
    def __init__(self, in_channels):
        super(GWF, self).__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xRGB, xIR = x[0], x[1]
        out = torch.cat([xRGB, xIR], dim=1)
        G = self.gate(out)

        PG = xRGB * G
        FG = xIR * (1 - G)

        return PG+FG


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class CLSP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return attn


class DFSC(nn.Module):
    def __init__(self, in_channels) -> None:
        super(DFSC, self).__init__()

        self.lsk = CLSP(in_channels)

        self.res_v = ResidualBlock(in_channels, in_channels)
        self.res_t = ResidualBlock(in_channels, in_channels)

    def forward(self, v_feat, t_feat):
        d_feat1 = v_feat - t_feat
        d_feat2 = t_feat - v_feat

        d_vector1 = self.lsk(d_feat1)
        d_vector2 = self.lsk(d_feat2)

        vd_feat = v_feat * d_vector1
        td_feat = t_feat * d_vector2

        v_feat_ = v_feat + td_feat
        t_feat_ = t_feat + vd_feat

        v_feat = v_feat + self.res_v(v_feat_)
        t_feat = t_feat + self.res_t(t_feat_)

        return v_feat, t_feat


@DETECTORS.register_module()
class Faster_RCNN_CFFDNet(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 dfsc=None,
                 gwf=None,
                 init_cfg=None):
        super(Faster_RCNN_CFFDNet, self).__init__(backbone)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        self.backbone_o = build_backbone(backbone)
        self.backbone_s = build_backbone(backbone)

        in_channels = neck['in_channels']

        if dfsc:
            self.DFSC = True
            self.DFSC_0 = DFSC(in_channels[0])
            self.DFSC_1 = DFSC(in_channels[1])
            self.DFSC_2 = DFSC(in_channels[2])
            self.DFSC_3 = DFSC(in_channels[3])
        else:
            self.DFSC = False

        if gwf:
            self.GWF = True
            self.GWF_0 = GWF(in_channels[0])
            self.GWF_1 = GWF(in_channels[1])
            self.GWF_2 = GWF(in_channels[2])
            self.GWF_3 = GWF(in_channels[3])
        else:
            self.GWF = False

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        o_img, s_img = img
        o_feat = self.backbone_o(o_img)
        s_feat = self.backbone_s(s_img)
        o_feat_0, o_feat_1, o_feat_2, o_feat_3 = o_feat[0], o_feat[1], o_feat[2], o_feat[3]
        s_feat_0, s_feat_1, s_feat_2, s_feat_3 = s_feat[0], s_feat[1], s_feat[2], s_feat[3]

        if self.DFSC:
            o_feat_0, o_feat_0 = self.DFSC_0(o_feat_0, s_feat_0)
            o_feat_1, s_feat_1 = self.DFSC_1(o_feat_1, s_feat_1)
            o_feat_2, s_feat_2 = self.DFSC_2(o_feat_2, s_feat_2)
            o_feat_3, s_feat_3 = self.DFSC_3(o_feat_3, s_feat_3)

        if self.GWF:
            x0, x1, x2 ,x3 = self.GWF_0([o_feat_0, s_feat_0]), self.GWF_1([o_feat_1, s_feat_1]), self.GWF_2([o_feat_2, s_feat_2]), self.GWF_3([o_feat_3, s_feat_3])
        else:
            x0, x1, x2, x3 = o_feat_0 + s_feat_0, o_feat_1 + s_feat_1, o_feat_2 + s_feat_2, o_feat_3 + s_feat_3

        x = [x0, x1, x2, x3]

        if self.with_neck:
            x = self.neck(x)
        return x


