from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=True
        )
        # self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=True
        )
        # self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class AlignmentModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        feature_channels: int = 60,
    ) -> None:
        super(AlignmentModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, feature_channels, kernel_size=3, padding=1, bias=True
        )

        self.att11 = nn.Conv2d(
            feature_channels * 2,
            feature_channels * 2,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.att12 = nn.Conv2d(
            feature_channels * 2, feature_channels, kernel_size=3, padding=1, bias=True
        )
        self.att31 = nn.Conv2d(
            feature_channels * 2,
            feature_channels * 2,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.att32 = nn.Conv2d(
            feature_channels * 2, feature_channels, kernel_size=3, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            feature_channels * 3, feature_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x1, x2, x3):
        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))

        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = F.sigmoid(F1_A)
        F1_ = F1_ * F1_A

        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.relu(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = F.sigmoid(F3_A)
        F3_ = F3_ * F3_A

        F_ = torch.cat((F1_, F2_, F3_), 1)

        F_0 = self.conv2(F_)
        return F_0


class ChannelGate(nn.Module):
    """Channel Attention from https://github.com/Jongchan/attention-module/"""

    def __init__(self, gate_channels, reduction_ratio=10, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class SelfAttention(nn.Module):
    """Self-Attention from https://github.com/swz30/Restormer"""

    def __init__(self, dim, num_heads, bias=True):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class CrossSelfAttention(nn.Module):
    def __init__(
        self, feature_channels: int, prior_dim: int, num_heads: int, bias: bool = True
    ) -> None:
        super(CrossSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=feature_channels,
            bias=bias,
        )
        self.kv = nn.Conv2d(prior_dim, prior_dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            prior_dim * 2,
            prior_dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=prior_dim * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            feature_channels, feature_channels, kernel_size=1, bias=bias
        )

    def forward(self, x, z):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))

        kv = self.kv_dwconv(self.kv(z))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


# LPENet
class LPENet(nn.Module):
    def __init__(
        self, in_channels, out_channels=3, feature_channels=60, num_residual_blocks=4
    ):
        super(LPENet, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels, feature_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.LeakyReLU(inplace=True),
        ]
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(feature_channels))
        layers.append(
            nn.Conv2d(
                feature_channels, out_channels, kernel_size=3, padding=1, bias=True
            )
        )
        self.lpe_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.lpe_net(x)


# Feature Refinement Module
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_channels, k=2):
        super(FeatureRefinementModule, self).__init__()
        self.rb1 = ResidualBlock(in_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=k)
        self.up = nn.Upsample(scale_factor=k, mode="nearest")
        self.norm = LayerNorm(in_channels, "BiasFree")
        self.attn = SelfAttention(in_channels, 6)
        self.rb2 = ResidualBlock(in_channels)
        self.conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=True)
        self.channel_gate = ChannelGate(in_channels, reduction_ratio=10)

    def forward(self, x):
        Fn_1 = self.rb1(x)

        Fn_1low = self.avgpool(Fn_1)
        Fn_1high = Fn_1 - self.up(Fn_1low)

        Fn_1low = self.attn(self.norm(Fn_1low)) + Fn_1low

        Fn_1high = self.rb2(Fn_1high)

        Fn_1cat = torch.cat([Fn_1high, self.up(Fn_1low)], 1)

        out = self.conv1(Fn_1cat)
        out = self.channel_gate(out)

        return out + x


# Prior Integration Module
class PriorIntegrationModule(nn.Module):
    def __init__(self, in_channels, prior_channels, k=4):
        super(PriorIntegrationModule, self).__init__()
        self.rb1 = ResidualBlock(in_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=k)
        self.up = nn.Upsample(scale_factor=k, mode="nearest")
        self.norm = LayerNorm(in_channels, "BiasFree")
        self.attn = CrossSelfAttention(in_channels, prior_channels, 3)
        self.rb2 = ResidualBlock(in_channels)
        self.conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=True)
        self.channel_gate = ChannelGate(in_channels, reduction_ratio=10)

    def forward(self, x, z):
        Fn_1 = self.rb1(x)

        Fn_1low = self.avgpool(Fn_1)
        Fn_1high = Fn_1 - self.up(Fn_1low)

        Fn_1low = self.attn(self.norm(Fn_1low), z) + Fn_1low

        Fn_1high = self.rb2(Fn_1high)

        Fn_1cat = torch.cat([Fn_1high, self.up(Fn_1low)], dim=1)

        out = self.conv1(Fn_1cat)
        out = self.channel_gate(out)

        return out + x


# DHRNet
class DHRNet(nn.Module):
    def __init__(self, in_channels, prior_channels, num_layers: List[int]) -> None:
        super(DHRNet, self).__init__()

        self.num_layers = num_layers

        for i in num_layers:
            self.add_module(
                f"pim_{i}",
                PriorIntegrationModule(
                    in_channels=in_channels, prior_channels=prior_channels, k=4
                ),
            )
            for j in range(i):
                self.add_module(
                    f"frm_{i}_{j}",
                    FeatureRefinementModule(in_channels=in_channels, k=2),
                )

    def forward(self, x, z):
        for i in self.num_layers:
            x = getattr(self, f"pim_{i}")(x, z)
            for j in range(i):
                x = getattr(self, f"frm_{i}_{j}")(x)
        return x
