"""Official ScribblePrompt UNet architecture.

Vendored from halleewong/ScribblePrompt commit
182c44975f77749b559974ce8db558c8bde57788 under Apache-2.0.
Only the inference network is included; training and demo dependencies are not.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        do_activation: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        ]
        if do_activation:
            layers.append(nn.PReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class _UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [64, 64, 64, 64, 64],
        conv_kernel_size: int = 3,
        conv: Optional[nn.Module] = None,
        conv_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        padding = (conv_kernel_size - 1) // 2
        self.in_channels = in_channels
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature_count in features:
            self.downs.append(
                conv(
                    current_channels,
                    feature_count,
                    kernel_size=conv_kernel_size,
                    padding=padding,
                    **conv_kwargs
                )
            )
            current_channels = feature_count

        for feature_count in reversed(features):
            self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.ups.append(
                conv(
                    feature_count * 2,
                    feature_count,
                    kernel_size=conv_kernel_size,
                    padding=padding,
                    **conv_kwargs
                )
            )

        self.bottleneck = conv(
            features[-1],
            features[-1],
            kernel_size=conv_kernel_size,
            padding=padding,
            **conv_kwargs
        )
        self.final_conv = conv(
            features[0],
            out_channels,
            kernel_size=1,
            padding=0,
            do_activation=False,
            **conv_kwargs
        )

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip = skip_connections[index // 2]
            x = self.ups[index + 1](torch.cat((skip, x), dim=1))
        return self.final_conv(x)


class UNet(_UNet):
    def __init__(self, **kwargs):
        super().__init__(conv=Conv2d, **kwargs)
