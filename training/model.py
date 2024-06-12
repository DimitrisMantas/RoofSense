from collections.abc import Iterable, Sequence
from typing import Literal

import torch.nn as nn
import torchseg.base
import torchseg.decoders.deeplabv3 as dlv3
import torchseg.decoders.deeplabv3.decoder as dlv3_decoder


class DeepLabV3Plus(dlv3.DeepLabV3Plus):
    def __init__(
        self,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        seperable: bool = False,
        attention: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            *args,
            **kwargs,
        )
        self.decoder.aspp[0] = ASPP(
            self.encoder.out_channels[-1],
            decoder_channels,
            decoder_atrous_rates,
            seperable,
            attention,
        )


class DeepLabV3PlusDecoder(dlv3_decoder.DeepLabV3PlusDecoder):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        out_channels: int,
        atrous_rates: Iterable[int],
        seperable: bool,
        attention: bool,
        output_stride: Literal[8, 16],
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
            encoder_channels[-1], out_channels, atrous_rates, seperable, attention
        )

# TODO: Add an SE module after each convolution stage.
class ASPP(dlv3_decoder.ASPP):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: Iterable[int],
        seperable: bool,
        attention: bool,
    ):
        """Atrous Spatial Pyramid Pooling module for the DeepLabv3(+) model architecture.

        References:
            https://arxiv.org/abs/1802.02611
            https://www.nature.com/articles/s41598-024-60375-1
        """
        super().__init__(in_channels, out_channels, atrous_rates, seperable)

        pyramid_in_channels = 5 * out_channels
        self.project = nn.Sequential(
            torchseg.base.Attention(
                name="scse" if attention else None,
                in_channels=pyramid_in_channels,
            ),
            nn.Conv2d(pyramid_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # TODO: See if this hinders training.
            nn.Dropout(0.5),
        )
