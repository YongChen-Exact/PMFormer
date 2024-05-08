from einops import rearrange
from torch import nn
import torch
import numpy as np
from pmformer.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_


class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()


class ConvDropoutNormNonlin(nn.Module):

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.GELU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs[
            'p'] > 0:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        self.lrelu = nonlin(**nonlin_kwargs) if nonlin_kwargs != None else nonlin()

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


# state channel attention
class Channel_Att(nn.Module):

    def __init__(self, k_size=3):
        super(Channel_Att, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.conv_avg = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_max = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mvg_result = self.avgpool(x)
        max_result = self.maxpool(x)
        mvg_out = self.conv_avg(mvg_result.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(
            -1).unsqueeze(-1)
        max_out = self.conv_max(max_result.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(
            -1).unsqueeze(-1)
        output = self.sigmoid(mvg_out + max_out)
        return x * output.expand_as(x)


# state spatial attention
class Spatial_Att(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(Spatial_Att, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv3d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm3d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv3d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                             kernel_size=3, \
                                             padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm3d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv3d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


# state channel and spatial attention
class CS_Att(nn.Module):

    def __init__(self, channel):
        super(CS_Att, self).__init__()
        self.channel_att = Channel_Att(k_size=3)
        self.spatial_att = Spatial_Att(channel)

    def forward(self, x):
        ca_out = self.channel_att(x)
        sa_out = self.spatial_att(x)
        att = 1 + F.sigmoid(ca_out + sa_out)
        return att * x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x


# state progressive guided fusion
class AdvancedTransformerBlock_PGF(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_PGF(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix, skip=None, x_up=None):
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"
        shortcut = x
        skip = skip.view(B, S, H, W, C)
        x_up = x_up.view(B, S, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size
        skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = skip.shape
        if self.shift_size > 0:
            skip = torch.roll(skip, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            x_up = torch.roll(x_up, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            skip = skip
            x_up = x_up
            attn_mask = None
        skip = window_partition(skip, self.window_size)
        skip = skip.view(-1, self.window_size * self.window_size * self.window_size,
                         C)
        x_up = window_partition(x_up, self.window_size)
        x_up = x_up.view(-1, self.window_size * self.window_size * self.window_size,
                         C)
        attn_windows = self.attn(skip, x_up, mask=attn_mask, pos_embed=None)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()
        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class WindowAttention_PGF(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, skip, x_up, pos_embed=None, mask=None):
        B_, N, C = skip.shape
        kv = self.kv(skip)
        q = x_up
        kv = kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k, v = kv[0], kv[1]
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pos_embed=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# state advanced transformer block
class AdvancedTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"
        shortcut = x
        x = x.view(B, S, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)
        attn_windows = self.attn(x_windows, mask=attn_mask, pos_embed=None)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()
        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim * 2)

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C)
        return x


class Patch_Expanding(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up = nn.ConvTranspose3d(dim, dim // 2, 2, 2)

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.up(x)
        x = ContiguousGrad.apply(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C // 2)
        return x


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 i_layer=None,
                 conv_op=None,
                 norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None,
                 basic_block=ConvDropoutNormNonlin
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # state params of LRE_block
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        conv_kwargs['kernel_size'] = conv_kernel_sizes[i_layer]
        conv_kwargs['padding'] = conv_pad_sizes[i_layer]

        # define parallel mixed module
        self.conv_blocks = nn.Sequential(
            *([basic_block(dim, dim, conv_op,
                           conv_kwargs,
                           norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                           nonlin, nonlin_kwargs)] +
              [basic_block(dim, dim, conv_op,
                           conv_kwargs,
                           norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                           nonlin, nonlin_kwargs)]))
        # define channel and spatial attention
        self.cs_att = CS_Att(channel=dim)

        # define advanced transformer block
        self.at_blocks = nn.ModuleList([
            AdvancedTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_m, S, H, W):
        x_l = x
        x_l = x_l.view(-1, S, H, W, x_l.size(-1)).permute(0, 4, 1, 2, 3).contiguous()
        x_l = self.conv_blocks(x_l)
        x_l = self.cs_att(x_l)
        x_l = rearrange(x_l, 'b c s h w -> b (s h w) c')
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.at_blocks:
            x = blk(x, attn_mask)
        x = x + x_l
        if self.downsample is not None:
            x_m = rearrange(x_m, 'b c s h w -> b (s h w) c')
            x_down = self.downsample(torch.cat((x_m, x), axis=2), S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W


class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        # build blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(
            AdvancedTransformerBlock_PGF(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
        )
        for i in range(depth - 1):
            self.blocks.append(
                AdvancedTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            )

        self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer)

    def forward(self, x, skip, S, H, W):
        x_up = self.Upsample(x, S, H, W)
        x = x_up + skip
        S, H, W = S * 2, H * 2, W * 2
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        x = self.blocks[0](x, attn_mask, skip=skip, x_up=x_up)
        for i in range(self.depth - 1):
            x = self.blocks[i + 1](x, attn_mask)
        return x, S, H, W


class project(nn.Module):

    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1 = [patch_size[0], patch_size[1] // 2, patch_size[2] // 2]
        stride2 = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
        self.proj1 = project(in_chans, embed_dim // 2, stride1, 1, nn.GELU, nn.LayerNorm, False)
        self.proj2 = project(embed_dim // 2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        x = self.proj1(x)
        x = self.proj2(x)
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)
        return x


class Encoder(nn.Module):

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 conv_op=None,
                 norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None
                 ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # define multi_scale pyramid
        self.scale_img1 = nn.AvgPool3d((2, 4, 4), (2, 4, 4))
        self.scale_img2 = nn.AvgPool3d((2, 2, 2), (2, 2, 2))
        self.scale_img3 = nn.AvgPool3d((2, 2, 2), (2, 2, 2))
        conv_kwargs = {'kernel_size': (3, 3, 3), 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer < 3:
                self.convs.append(ConvDropoutNormNonlin(in_chans, int(embed_dim * 2 ** i_layer), conv_op,
                                                        conv_kwargs,
                                                        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                                        nonlin, nonlin_kwargs))
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
                i_layer=i_layer,
                conv_op=conv_op,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes
            )
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        x_raw = x
        x = self.patch_embed(x)
        down = []
        mul_imgs = []

        scale_img_1 = self.scale_img1(x_raw)
        scale_img_conv1 = self.convs[0](scale_img_1)
        mul_imgs.append(scale_img_conv1)

        scale_img_2 = self.scale_img2(scale_img_1)
        scale_img_conv2 = self.convs[1](scale_img_2)
        mul_imgs.append(scale_img_conv2)

        scale_img_3 = self.scale_img3(scale_img_2)
        scale_img_conv3 = self.convs[2](scale_img_3)
        mul_imgs.append(scale_img_conv3)

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            if i < 3:
                x_out, S, H, W, x, Ws, Wh, Ww = layer(x, mul_imgs[i], Ws, Wh, Ww)
            else:
                x_out, S, H, W, x, Ws, Wh, Ww = layer(x, None, Ws, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                down.append(out)
        return down


class Decoder(nn.Module):

    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 depths=[2, 2, 2],
                 num_heads=[24, 12, 6],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[2] // patch_size[2] // 2 ** (len(depths) - i_layer - 1)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        outs = []
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        for index, i in enumerate(skips):
            i = i.flatten(2).transpose(1, 2).contiguous()
            skips[index] = i
        x = self.pos_drop(x)
        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            x, S, H, W, = layer(x, skips[i], S, H, W)
            out = x.view(-1, S, H, W, self.num_features[i])
            outs.append(out)
        return outs


class final_patch_expanding(nn.Module):

    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.up = nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.up(x)
        return x


class PMFormer(SegmentationNetwork):

    def __init__(self, crop_size=[64, 128, 128],
                 embedding_dim=192,
                 input_channels=1,
                 num_classes=2,
                 conv_kernel_sizes=None,
                 conv_op=nn.Conv3d,
                 depths=[2, 2, 2, 2],
                 num_heads=[6, 12, 24, 48],
                 patch_size=[2, 4, 4],
                 window_size=[4, 4, 8, 4],
                 deep_supervision=True):
        super(PMFormer, self).__init__()
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.upscale_logits_ops = []
        self.upscale_logits_ops.append(self.TMP_func)
        embed_dim = embedding_dim
        depths = depths
        num_heads = num_heads
        patch_size = patch_size
        window_size = window_size
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op = nn.Dropout3d
        dropout_op_kwargs = {'p': 0.1, 'inplace': True}
        nonlin = nn.GELU
        nonlin_kwargs = None
        conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        self.model_down = Encoder(pretrain_img_size=crop_size, window_size=window_size, embed_dim=embed_dim,
                                  patch_size=patch_size, depths=depths, num_heads=num_heads, in_chans=input_channels,
                                  conv_op=self.conv_op, norm_op=norm_op,
                                  norm_op_kwargs=norm_op_kwargs,
                                  dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin,
                                  nonlin_kwargs=nonlin_kwargs,
                                  conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes)
        self.decoder = Decoder(pretrain_img_size=crop_size, embed_dim=embed_dim, window_size=window_size[::-1][1:],
                               patch_size=patch_size, num_heads=num_heads[::-1][1:], depths=depths[::-1][1:])
        self.final = []
        if self.do_ds:
            for i in range(len(depths) - 1):
                if i == 2:
                    self.final.append(final_patch_expanding(embed_dim * 2 ** i, num_classes, patch_size=(2, 4, 4)))
                elif i == 1:
                    self.final.append(final_patch_expanding(embed_dim * 2 ** i, num_classes, patch_size=(2, 4, 4)))
                else:
                    self.final.append(final_patch_expanding(embed_dim * 2 ** i, num_classes, patch_size=patch_size))
        else:
            self.final.append(final_patch_expanding(embed_dim, num_classes, patch_size=patch_size))
        self.final = nn.ModuleList(self.final)

    def TMP_func(self, x):
        return x

    def forward(self, x):
        seg_outputs = []
        skips = self.model_down(x)
        neck = skips[-1]
        out = self.decoder(neck, skips)
        if self.do_ds:
            for i in range(len(out)):
                seg_outputs.append(self.final[-(i + 1)](out[i]))
            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs[-1]
