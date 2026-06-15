# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""GLS-YOLO modules for UAV poppy detection.

This file contains the custom modules described in the GLS-YOLO paper:
GRAConv, LGCM, and SEAM. They are kept inside the Ultralytics module
namespace so YAML model definitions can reference them directly.
"""

import torch
import torch.nn as nn

__all__ = "GRAConv", "LGCM", "SEAM", "MultiSEAM"


def _to_int_tuple(v):
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return int(v)


def autopad(k, p=None, d=1):
    """Pad to same shape outputs."""
    k = _to_int_tuple(k)
    d = _to_int_tuple(d)
    if isinstance(d, tuple):
        d = d[0]
    if isinstance(k, tuple):
        ek = (d * (k[0] - 1) + 1, d * (k[1] - 1) + 1) if d > 1 else k
        return (ek[0] // 2, ek[1] // 2) if p is None else _to_int_tuple(p)
    ek = d * (k - 1) + 1 if d > 1 else k
    return ek // 2 if p is None else int(p)


class GLSConv(nn.Module):
    """Standard Conv-BN-activation block used by GLS-YOLO modules."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__()
        c1, c2 = int(c1), int(c2)
        self.conv = nn.Conv2d(c1, c2, k, s, padding=autopad(k, d=d), dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GRAConv(nn.Module):
    """Gradient-Region Aggregation Convolution.

    The module compares local and dilated contextual responses and uses their
    difference to generate a structural attention map for downsampling stages.
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c1, c2, k, s = int(c1), int(c2), int(k), int(s)
        hidden = max(c2 // 2, 32)
        self.reduce = GLSConv(c1, hidden, k=1, s=1)
        self.local_conv = GLSConv(hidden, hidden, k=k, s=s, d=1)
        self.context_conv = GLSConv(hidden, hidden, k=k, s=s, d=2)
        self.diff_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
        )
        self.fuse = GLSConv(hidden, c2, k=1, s=1)
        self.proj = GLSConv(c1, c2, k=1, s=s, act=False) if s != 1 or c1 != c2 else None
        self.alpha = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))

    def forward(self, x):
        xr = self.reduce(x)
        y_local = self.local_conv(xr)
        y_context = self.context_conv(xr)
        diff = torch.abs(y_local - y_context)
        att = torch.sigmoid(self.diff_conv(diff))
        y = self.fuse(y_local + att * y_context)
        residual = self.proj(x) if self.proj is not None else x
        return residual + self.alpha * y


class _AntiAliasDW(nn.Module):
    """Depthwise anti-aliasing filter used by LGCM when stride is greater than 1."""

    def __init__(self, c, enable=True):
        super().__init__()
        self.enable = bool(enable)
        if self.enable:
            k = 3
            self.blur = nn.Conv2d(int(c), int(c), k, 1, k // 2, groups=int(c), bias=False)
            with torch.no_grad():
                self.blur.weight.data.fill_(1.0 / (k * k))
        else:
            self.blur = nn.Identity()

    def forward(self, x):
        return self.blur(x) if self.enable else x


class LGCM(nn.Module):
    """Lightweight Global-Local Context Modulator."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c1, c2 = int(c1), int(c2)
        self.k = _to_int_tuple(k)
        self.s = int(s)
        self.g = int(g)
        self.d = _to_int_tuple(d)
        if (isinstance(self.k, int) and self.k == 1) or (isinstance(self.k, tuple) and self.k == (1, 1)) or self.g > 1:
            pad = autopad(self.k, p, self.d)
            self.main = nn.Sequential(
                nn.Conv2d(c1, c2, self.k, self.s, pad, groups=self.g, dilation=self.d, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True) if act is True else act if isinstance(act, nn.Module) else nn.Identity(),
            )
            self.mode = "fallback"
            return
        self.mode = "lgcm"
        self.act = nn.SiLU(inplace=True) if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.antiblur = _AntiAliasDW(c1, enable=self.s > 1)
        base_d = self.d if isinstance(self.d, int) else self.d[0]
        base_d = max(1, int(base_d))
        self.dw_local = nn.Conv2d(c1, c1, self.k, self.s, autopad(self.k, p, base_d), groups=c1, dilation=base_d, bias=False)
        self.bn_local = nn.BatchNorm2d(c1)
        ctx_d = max(1, 2 * base_d)
        self.dw_ctx = nn.Conv2d(c1, c1, self.k, self.s, autopad(self.k, p, ctx_d), groups=c1, dilation=ctx_d, bias=False)
        self.bn_ctx = nn.BatchNorm2d(c1)
        kk = self.k if isinstance(self.k, int) else self.k[0]
        mid = max(c1 // 4, 8)
        self.ctx_pool = nn.AvgPool2d(kernel_size=int(kk), stride=self.s, padding=int(kk) // 2)
        self.gate_mlp = nn.Sequential(nn.Conv2d(c1, mid, 1), nn.ReLU(inplace=True), nn.Conv2d(mid, 1, 1))
        self.pw_out = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(c2)
        self.res_scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):
        if self.mode == "fallback":
            return self.main(x)
        xin = self.antiblur(x) if self.s > 1 else x
        y_local = self.bn_local(self.dw_local(xin))
        y_context = self.bn_ctx(self.dw_ctx(xin))
        gate = torch.sigmoid(self.gate_mlp(self.ctx_pool(xin)))
        y_mix = (1.0 - gate) * y_local + gate * y_context
        out = self.act(self.bn_out(self.pw_out(y_mix)))
        if x.shape[1] == out.shape[1] and x.shape[-2:] == out.shape[-2:]:
            return x + self.res_scale * out
        return self.res_scale * out


class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def _dcovn(c1, c2, depth, kernel_size=3, patch_size=3):
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[
            nn.Sequential(
                _Residual(
                    nn.Sequential(
                        nn.Conv2d(c2, c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                        nn.SiLU(),
                        nn.BatchNorm2d(c2),
                    )
                ),
                nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0),
                nn.SiLU(),
                nn.BatchNorm2d(c2),
            )
            for _ in range(depth)
        ],
    )


class SEAM(nn.Module):
    """Separated and Enhancement Attention Module."""

    def __init__(self, c1, n=1, reduction=16):
        super().__init__()
        c2 = int(c1)
        hidden = max(c2 // int(reduction), 1)
        self.dcovn = nn.Sequential(
            *[
                nn.Sequential(
                    _Residual(
                        nn.Sequential(
                            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2),
                            nn.GELU(),
                            nn.BatchNorm2d(c2),
                        )
                    ),
                    nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0),
                    nn.GELU(),
                    nn.BatchNorm2d(c2),
                )
                for _ in range(int(n))
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c2, hidden, bias=False), nn.ReLU(inplace=True), nn.Linear(hidden, c2, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.dcovn(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * torch.exp(y).expand_as(x)


class MultiSEAM(nn.Module):
    """Multi-scale SEAM variant using several patch embeddings."""

    def __init__(self, c1, depth=1, kernel_size=3, patch_size=(6, 7, 8), reduction=16):
        super().__init__()
        c2 = int(c1)
        hidden = max(c2 // int(reduction), 1)
        self.branches = nn.ModuleList([_dcovn(c2, c2, depth, kernel_size=kernel_size, patch_size=p) for p in patch_size])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c2, hidden, bias=False), nn.ReLU(inplace=True), nn.Linear(hidden, c2, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        pooled = [self.avg_pool(branch(x)).view(b, c) for branch in self.branches]
        pooled.append(self.avg_pool(x).view(b, c))
        y = sum(pooled) / len(pooled)
        y = self.fc(y).view(b, c, 1, 1)
        return x * torch.exp(y).expand_as(x)
