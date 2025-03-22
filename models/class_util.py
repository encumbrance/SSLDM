import torch as th
from torch import nn
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    zero_module,
)

from torchvision.ops import DeformConv2d

from ldm.modules.diffusionmodules.openaimodel import Upsample,  AttentionBlock

class GroupNorm4(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm4(4, channels)

class dense_block(nn.Module):
    def __init__(self, hint_channels, model_channels, dims, ckpt_path=None):
        super(dense_block, self).__init__()
        self.input_hint_block = nn.Sequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            # conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            conv_nd(dims, 16, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

    def forward(self, x):
        return self.input_hint_block(x)

        

class OriResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param out_channels: if specified, the number of output channels.
    :param dropout: the rate of dropout.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.updown = up or down
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size=3, padding=1),
        )

        if up:
            self.h_upd = Upsample(channels, use_conv=False, dims=dims)
            self.x_upd = Upsample(channels, use_conv=False, dims=dims)
        elif down:
            self.h_upd = Downsample(channels, use_conv=False, dims=dims)
            self.x_upd = Downsample(channels, use_conv=False, dims=dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size=3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size=3, padding=1
            )
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size=1
            )

    def forward(self, x):
        """
        Apply the residual block to the input tensor.
        :param x: input tensor of shape (B, C, H, W) or (B, C, D, H, W).
        :return: output tensor of the same shape as input.
        """
        assert x.shape[1] == self.channels
        h = self.in_layers(x)
        h = self.h_upd(h)
        x = self.x_upd(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class deform_conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=False, padding=1, stride=1):
        super(deform_conv_2d, self).__init__()
        self.groups = groups
        self.conv_offset = conv_nd(2, in_channels, 2*self.groups*kernel_size*kernel_size, kernel_size, bias=bias, padding=padding, stride=stride)
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size, groups=self.groups, bias=bias, padding=padding, stride=stride)
        

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.conv(x, offset)
    

class DeformResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param out_channels: if specified, the number of output channels.
    :param dropout: the rate
    """
    def __init__(
        self,
        channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.updown = up or down
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            deform_conv_2d(channels, self.out_channels, kernel_size=3),
        )

        if up:
            self.h_upd = Upsample(channels, use_conv=False, dims=dims)
            self.x_upd = Upsample(channels, use_conv=False, dims=dims)
        elif down:
            self.h_upd = Downsample(channels, use_conv=False, dims=dims)
            self.x_upd = Downsample(channels, use_conv=False, dims=dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                deform_conv_2d(self.out_channels, self.out_channels, kernel_size=3)
            ),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()    
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        """
        Apply the residual block to the input tensor.
        :param x: input tensor of shape (B, C, H, W) or (B, C, D, H, W).
        :return: output tensor of the same shape as input.
        """
        assert x.shape[1] == self.channels
        h = self.in_layers(x)
        h = self.h_upd(h)
        x = self.x_upd(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

def max_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D max pooling module.
    """
    if dims == 1:
        return nn.MaxPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.MaxPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.MaxPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = max_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    

class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_new_attention_order=False,
        legacy=True,
        ckpt_path=None, 
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample


        self.input_blocks = nn.ModuleList(
            [conv_nd(dims, in_channels, model_channels, 3, padding=1)]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    OriResBlock(
                        ch,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        dims=dims,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) 
                    )

                
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = num_head_channels
        self.middle_block = nn.Sequential(
            OriResBlock(
                ch,
                out_channels=ch,
                dropout=dropout,
                dims=dims,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ),
            OriResBlock(
                ch,
                out_channels=ch,
                dropout=dropout,
                dims=dims,
            ),
        )
 
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    OriResBlock(
                        ch + ich,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        dims=dims,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch


     
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, self.num_classes, 3, padding=1)),
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, ckpt_path):
        sd = th.load(ckpt_path)
        self.load_state_dict(sd)
    
    def forward(self, x):
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h)
        h = h.type(x.dtype)
        return self.out(h)
