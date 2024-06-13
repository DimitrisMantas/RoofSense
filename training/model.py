from collections.abc import Iterable, Sequence
from typing import Literal

import torch.nn as nn
import torchseg.base
import torchseg.decoders.deeplabv3 as dlv3
import torchseg.decoders.deeplabv3.decoder as dlv3_decoder


class DeepLabV3Plus(dlv3.DeepLabV3Plus):
    def __init__(
        self,
        encoder_output_stride: Literal[8, 16] = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        separable: bool = True,
        attention: bool = False,
        dropout: float | None = 0.5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            *args,
            **kwargs,
        )
        self.decoder = DeepLabV3PlusDecoder(
            self.encoder.out_channels,
            decoder_channels,
            decoder_atrous_rates,
            separable,
            attention,
            encoder_output_stride,
            dropout,
        )


class DeepLabV3PlusDecoder(dlv3_decoder.DeepLabV3PlusDecoder):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        out_channels: int,
        atrous_rates: Iterable[int],
        separable: bool,
        attention: bool,
        output_stride: Literal[8, 16],
        dropout: float | None,
    ) -> None:
        """DeepLabv3+ decoder.

        Args:
            encoder_channels:
                The number of output channels in each encoder stage. The last output
                defines the input size of the decoder along the corresponding dimension.

        References:
            https://arxiv.org/abs/1802.02611
            https://www.nature.com/articles/s41598-024-60375-1


        """
        super().__init__(encoder_channels, out_channels, atrous_rates, output_stride)
        self.aspp[0] = ASPP(
            encoder_channels[-1],
            out_channels,
            atrous_rates,
            separable,
            attention,
            dropout,
        )


# TODO: Add an SE module after each convolution stage.
class ASPP(dlv3_decoder.ASPP):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: Iterable[int],
        separable: bool,
        attention: bool,
        dropout: float | None,
    ):
        """Atrous Spatial Pyramid Pooling module for the DeepLabv3(+) model architecture.

        References:
            https://arxiv.org/abs/1802.02611
            https://www.nature.com/articles/s41598-024-60375-1
        """
        super().__init__(in_channels, out_channels, atrous_rates, separable)

        self.convs[-1] = ASPPPooling(in_channels, out_channels)

        pyramid_in_channels = 5 * out_channels
        self.project = nn.Sequential()
        if attention:
            self.project.append(
                torchseg.base.Attention(name="scse", in_channels=pyramid_in_channels)
            )
        self.project.append(
            nn.Conv2d(pyramid_in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.project.append(nn.BatchNorm2d(out_channels))
        self.project.append(nn.ReLU())
        if dropout is not None:
            self.project.append(nn.Dropout(dropout))


class ASPPPooling(dlv3_decoder.ASPPPooling):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels, out_channels)
        self[0] = nn.AdaptiveMaxPool2d(1)
