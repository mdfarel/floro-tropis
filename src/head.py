"""UPerNet segmentation head on frozen, cached ViT features (PANGAEA frozen-encoder protocol, sized for CPU).

Input:  (B, L, C, G, G) features from L = 4 encoder layers (C channels on a G x G token grid; C and G are read
        from the data, e.g. C = 1024, G = 16 for 256 px input with patch 16, or C = 768/1024, G = 14).
Output: (B, num_classes, out_size, out_size) logits, bilinearly upsampled (default 2 classes, 256 x 256).

Architecture (mirrors PANGAEA `SegUPerNet` = Feature2Pyramid neck + mmseg UPerHead):
  neck, one branch per encoder layer (layer order shallow -> deep):
    layer 0: x4 upsample  deconv(C->C, k2 s2) - BN - GELU - deconv(k2 s2)        -> 4G x 4G
    layer 1: x2 upsample  deconv(k2 s2)                                           -> 2G x 2G
    layer 2: identity                                                             ->  G x  G
    layer 3: /2           maxpool(2)                                              -> G/2 x G/2
  UPerHead with `channels` decoder channels (PANGAEA default 512; here 128 for CPU):
    PPM on layer 3 (pool scales 1, 2, 3, 6; conv1x1-BN-ReLU each) + bottleneck conv3x3-BN-ReLU,
    lateral conv1x1-BN-ReLU on layers 0..2, top-down FPN (bilinear upsample + add), fpn conv3x3-BN-ReLU,
    concat of 4 levels at 4G x 4G, fpn_bottleneck conv3x3-BN-ReLU, Dropout2d(0.1), conv1x1 -> classes,
    bilinear upsample to out_size.

neck="fused" (default): in PANGAEA the last deconv of the x4 and x2 branches (C -> C) is followed directly by the
lateral conv1x1 (C -> channels, no bias). Two linear maps in a row are one linear map, so here the last deconv of
each of those branches maps C -> channels directly and its lateral keeps only BN-ReLU. The set of functions the head
can represent is identical; the head is 1.9x smaller (8.7 M vs 16.3 M parameters at C = 1024) and about 3x cheaper per
training step on CPU (measured at C = 1024, G = 16, batch 16, 3 threads: 2.1 s vs 6.3 s), because the full-width
C x C deconvs at 2G and 4G resolution dominate the cost otherwise. neck="pangaea" keeps the literal C -> C deconvs plus
lateral conv1x1. Convs before BatchNorm have no bias (mmcv ConvModule default; any such bias is cancelled by the
BatchNorm), and BatchNorm2d stands in for SyncBatchNorm (identical in one process).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F


def _up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ConvBNReLU(nn.Sequential):
    """mmcv ConvModule default: conv (no bias) + BatchNorm + ReLU."""

    def __init__(self, cin, cout, k=1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class BNReLU(nn.Sequential):
    def __init__(self, c):
        super().__init__(nn.BatchNorm2d(c), nn.ReLU(inplace=True))


class UPerNetHead(nn.Module):
    def __init__(self, in_channels, num_layers=4, channels=128, num_classes=2, out_size=256,
                 pool_scales=(1, 2, 3, 6), neck="fused", dropout=0.1):
        super().__init__()
        if num_layers != 4:
            raise ValueError("Feature2Pyramid neck expects 4 encoder layers (rescales 4, 2, 1, 0.5)")
        if neck not in ("fused", "pangaea"):
            raise ValueError("neck must be 'fused' or 'pangaea'")
        C, ch = in_channels, channels
        self.in_channels, self.channels, self.num_classes = C, ch, num_classes
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else tuple(out_size)
        self.neck_mode = neck
        last = ch if neck == "fused" else C

        # Feature2Pyramid neck (rescales 4, 2, 1, 0.5)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=2, stride=2),
            nn.BatchNorm2d(C),
            nn.GELU(),
            nn.ConvTranspose2d(C, last, kernel_size=2, stride=2),
        )
        self.up2 = nn.ConvTranspose2d(C, last, kernel_size=2, stride=2)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # UPerHead: laterals for levels 0..2 (level 3 goes through the PPM)
        if neck == "fused":
            self.lateral = nn.ModuleList([BNReLU(ch), BNReLU(ch), ConvBNReLU(C, ch, 1)])
        else:
            self.lateral = nn.ModuleList([ConvBNReLU(C, ch, 1) for _ in range(3)])
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), ConvBNReLU(C, ch, 1)) for s in pool_scales])
        self.psp_bottleneck = ConvBNReLU(C + len(pool_scales) * ch, ch, 3)
        self.fpn_convs = nn.ModuleList([ConvBNReLU(ch, ch, 3) for _ in range(3)])
        self.fpn_bottleneck = ConvBNReLU(4 * ch, ch, 3)
        self.dropout = nn.Dropout2d(dropout)
        self.conv_seg = nn.Conv2d(ch, num_classes, kernel_size=1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _psp(self, x):
        outs = [x] + [_up(m(x), x.shape[2:]) for m in self.ppm]
        return self.psp_bottleneck(torch.cat(outs, dim=1))

    def forward(self, feats):
        """feats: (B, L, C, G, G) tensor or a list of 4 (B, C, G, G) tensors (shallow -> deep)."""
        if torch.is_tensor(feats):
            feats = feats.unbind(1)
        f0, f1, f2, f3 = [f.float() for f in feats]
        lat = [
            self.lateral[0](self.up4(f0)),
            self.lateral[1](self.up2(f1)),
            self.lateral[2](f2),
            self._psp(self.down2(f3)),
        ]
        for i in range(3, 0, -1):  # top-down pathway
            lat[i - 1] = lat[i - 1] + _up(lat[i], lat[i - 1].shape[2:])
        outs = [self.fpn_convs[i](lat[i]) for i in range(3)] + [lat[3]]
        size = outs[0].shape[2:]
        x = torch.cat([outs[0]] + [_up(o, size) for o in outs[1:]], dim=1)
        x = self.fpn_bottleneck(x)
        x = self.conv_seg(self.dropout(x))
        return _up(x, self.out_size)

    def describe(self):
        return (f"UPerNet (PANGAEA SegUPerNet layout) on 4 frozen layers; Feature2Pyramid neck x4/x2/x1/maxpool; "
                f"decoder channels {self.channels} (PANGAEA default 512); neck={self.neck_mode}; "
                f"PPM scales 1,2,3,6; BN; Dropout2d 0.1; bilinear upsample to {self.out_size[0]}x{self.out_size[1]}; "
                f"{self.num_classes} classes; in_channels {self.in_channels}")


if __name__ == "__main__":
    torch.set_num_threads(3)
    for C, G in [(1024, 16), (768, 14), (1024, 14)]:
        for neck in ("fused", "pangaea"):
            h = UPerNetHead(C, neck=neck).eval()
            with torch.inference_mode():
                y = h(torch.randn(2, 4, C, G, G))
            print(f"C={C} G={G} neck={neck}: out {tuple(y.shape)}  params {h.n_params():,}")
