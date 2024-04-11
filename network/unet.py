import torch
import torch.nn as nn
from typing import Tuple

'''
https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising/blob/master-with-report/ssdn/ssdn/models/noise_network.py
'''
class Unet(nn.Module):
    """Custom U-Net architecture for Self Supervised Denoising (SSDN) and Noise2Noise (N2N).
    Base N2N implementation was made with reference to @joeylitalien's N2N implementation.
    Changes made are removal of weight sharing when blocks are reused. Usage of LeakyReLu
    over standard ReLu and incorporation of blindspot functionality.

    Unlike other typical U-Net implementations dropout is not used when the model is trained.

    When in blindspot mode the following behaviour changes occur:

        * Input batches are duplicated for rotations: 0, 90, 180, 270. This increases the
          batch size by 4x. After the encode-decode stage the rotations are undone and
          concatenated on the channel axis with the associated original image. This 4x
          increase in channel count is collapsed to the standard channel count in the
          first 1x1 kernel convolution.

        * To restrict the receptive field into the upward direction a shift is used for
          convolutions (see nn.Conv2d) and downsampling. Downsampling uses a single
          pixel shift prior to max pooling as dictated by Laine et al. This is equivalent
           to applying a shift on the upsample.

    Args:
        in_channels (int, optional): Number of input channels, this will typically be either
            1 (Mono) or 3 (RGB) but can be more. Defaults to 3.
        out_channels (int, optional): Number of channels the final convolution should output.
            Defaults to 3.
        blindspot (bool, optional): Whether to enable the network blindspot. This will
            add in rotation stages and shift stages while max pooling and during convolutions.
            A futher shift will occur after upsample. Defaults to False.
        zero_output_weights (bool, optional): Whether to initialise the weights of
                `nin_c` to zero. This is not mentioned in literature but is done as part
                of the tensorflow implementation for the parameter estimation network.
                Defaults to False.
    """

    def __init__(self, in_ch=3, out_ch=3, zero_output=False, dim=48):
        super(Unet, self).__init__()
        self.zero_output = zero_output
        in_channels = in_ch
        out_channels = out_ch

        ####################################
        # Encode Blocks
        ####################################

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.MaxPool2d(2)
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        self.encode_block_4 = _encode_block_2_3_4_5()
        self.encode_block_5 = _encode_block_2_3_4_5()

        # Layers: enc_conv6
        self.encode_block_6 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(dim * 3, dim * 2, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        # Separate instances of same decode module definition created
        self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            nn.Conv2d(dim * 2 + in_channels, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Output Block
        ####################################


        # nin_a,b,c, linear_act
        self.output_conv = nn.Conv2d(dim * 2, out_channels, 1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).

        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()

        # Initialise last output layer
        if self.zero_output:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward_train(self, x):

        # Encoder
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        # Decoder
        upsample5 = self.decode_block_6(encoded)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)

        x = self.output_conv(x)

        return x

    def forward_test(self, x):
        n, c, h, w = x.shape
        if h < w:
            x = torch.nn.functional.pad(x, [0, 0, 0, w - h], mode='reflect')
        else:
            x = torch.nn.functional.pad(x, [0, h - w, 0, 0], mode='reflect')
        x = self.forward_train(x)
        x = x[:, :, :h, :w]
        return x

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)

    @staticmethod
    def input_wh_mul() -> int:
        """Multiple that both the width and height dimensions of an input must be to be
        processed by the network. This is devised from the number of pooling layers that
        reduce the input size.

        Returns:
            int: Dimension multiplier
        """
        max_pool_layers = 5
        return 2 ** max_pool_layers