from tinygrad import nn, Tensor


class DoubleConv:
    def __init__(self, in_ch: int, out_ch: int):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding="same", bias=False)
        self.norm1 = nn.InstanceNorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding="same", bias=False)
        self.norm2 = nn.InstanceNorm(out_ch)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = x.leaky_relu()
        z = self.conv2(x)
        z = self.norm2(z)
        out = (z + x).leaky_relu()
        return out



class Up:
    def __init__(self, in_ch: int, ch: int, out_ch: int):
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.body = DoubleConv(ch, out_ch)

    def __call__(self, x: Tensor, skip: Tensor) -> Tensor:

        x = self.up(x)
        x = Tensor.cat(x, skip, dim=1)
        x = self.body(x)
        return x


class Network:
    def __init__(self, in_channels: int = 384, num_classes: int = 2):
        # 384->256 1/16->1/16
        self.bottleneck = DoubleConv(in_channels, 256)
        # 256->128 1/16->1/8
        self.up1 = Up(256, 128+64, 128)
        # 128->64 1/8->1/4
        self.up2 = Up(128, 64+16, 64)
        # 64->32 1/4->1/2
        self.up3 = Up(64, 32+4, 32)
        # 32->16 1/2->1
        self.up4 = Up(32, 16+1, 16)
        self.outc = nn.Conv2d(16, num_classes, 1)

    def __call__(self, e: Tensor,  s: Tensor) -> Tensor:
        # e.shape B, 384, 16, 16; x.shape B, 1, 512, 512
        B, C, H, W = s.shape
        # s8.shape B, 8*8*C, H // 8, W // 8; 64, 64, 64
        s8 = s.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, -1, H // 8, W // 8)
        # s4.shape B, 4*4*C, H // 4, W // 4; 16, 128, 128
        s4 = s.reshape(B, C, H // 4, 4, W // 4, 4).permute(0, 1, 3, 5, 2, 4).reshape(B, -1, H // 4, W // 4)
        # s2.shape B, 2*2*C, H // 2, W // 2; 4, 256, 256
        s2 = s.reshape(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, -1, H // 2, W // 2)
        # s1.shape B, C, H, W; 1, 512, 512
        s1 = s
        # e.shape B, 384, 32, 32
        # print(s8.shape, s4.shape, s2.shape, s1.shape, e.shape, 'call')
        e = self.bottleneck(e)
        x = self.up1(e, s8)
        x = self.up2(x, s4)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        x = self.outc(x)
        return x
