import torch.nn as nn
from convnext import ConvNextBlock


class DownsampleLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1):
        super(DownsampleLayer, self).__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size, 
                              stride=stride)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class UpsampleLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1):
        super(UpsampleLayer, self).__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose2d(in_channels, 
                                       out_channels, 
                                       kernel_size, 
                                       stride=stride,
                                       padding=kernel_size // 2,
                                       output_padding=stride - 1)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    

class EncoderBlock(nn.Module):

    def __init__(self, n_in, n_out, *, stride=1, down_kernel_size=None, n_convs=1):
        super(EncoderBlock, self).__init__()

        self.convs = nn.Sequential(*[
            ConvNextBlock(n_in) for _ in range(n_convs)
        ])
        
        self.down_conv = None
        if stride > 1:
            assert isinstance(down_kernel_size, int)
            self.down_conv = DownsampleLayer(
                n_in, n_out, down_kernel_size, stride=stride)
        else:
            assert n_in == n_out


    def forward(self, x):
        x = self.convs(x)
        if self.down_conv:
            x = self.down_conv(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, n_in, n_out, *, stride=1, up_kernel_size=None, n_convs=1):
        super(DecoderBlock, self).__init__()

        self.convs = nn.Sequential(*[
            ConvNextBlock(n_out) for _ in range(n_convs)
        ])
        
        self.up_conv = None
        if stride > 1:
            assert isinstance(up_kernel_size, int)
            self.up_conv = UpsampleLayer(
                n_in, n_out, up_kernel_size, stride=stride)
        else:
            assert n_in == n_out


    def forward(self, x):
        if self.up_conv:
            x = self.up_conv(x)
        x = self.convs(x)
        return x
    
    
class LatentDecoder(nn.Module):
    def __init__(
            self, 
            input_dim: int,
            hidden_dim: int,
            output_dim: int, 
            *, 
            strides: list[int], 
            up_kernels: list[int], 
            n_convs: list[int]
        ):
        super(LatentDecoder, self).__init__()

        self.decoder = nn.ModuleList()
        for i in range(len(strides)):
            self.decoder.append(
                DecoderBlock(
                    n_in=hidden_dim,
                    n_out=hidden_dim,
                    stride=strides[i],
                    up_kernel_size=up_kernels[i],
                    n_convs=n_convs[i]
                )
            )

        self.in_linear = nn.Conv2d(in_channels=input_dim,
                                 out_channels=hidden_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)


        self.out_linear = nn.Conv2d(in_channels=hidden_dim,
                                    out_channels=output_dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

    def forward(self, x):
        x = self.in_linear(x)
        for block in self.decoder:
            x = block(x)
        x = self.out_linear(x)
        return x