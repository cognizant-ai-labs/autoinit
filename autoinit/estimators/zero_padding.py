
# Copyright (C) 2021 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# AutoInit Software in commercial settings.
#
# END COPYRIGHT
from typing import List
from numpy import prod

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from framework.enn.autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator

CHANNELS_LAST = 'channels_last'

class ZeroPaddingOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of ZeroPadding layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        The 1D, 2D, and 3D variants of ZeroPadding layers add zeros to the borders of
        the input tensor.  Since the tensor is padded with zeros, we can calculate what
        fraction of the tensor the zeros occupy, and then use the same analysis to estimate
        the outgoing mean and variance as we did for Dropout layers.  Note: ZeroPadding is
        NOT the same as Dropout, but if we have an "effective_dropout_rate," we can calculate:

        mean_out = mean_in * (1.0 - effective_dropout_rate)
        var_out = var_in * (1.0 - effective_dropout_rate)

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """
        input_shape = self.layer.input_shape[1:] # ignore batch dimension
        padding = self.layer.padding
        data_format = self.layer.data_format

        if isinstance(self.layer, tfkeras.layers.ZeroPadding1D):
            output_shape = self._get_output_shape_1d(padding, input_shape)
        elif isinstance(self.layer, tfkeras.layers.ZeroPadding2D):
            output_shape = self._get_output_shape_2d(padding, input_shape, data_format)
        else:
            if not isinstance(self.layer, tfkeras.layers.ZeroPadding3D):
                raise NotImplementedError(f'Layer {type(self.layer).__name__} is not supported yet')
            output_shape = self._get_output_shape_3d(padding, input_shape, data_format)

        input_size = prod(input_shape)
        output_size = prod(output_shape)
        effective_dropout_rate = (output_size - input_size) / output_size
        mean_out = means_in[0] * (1.0 - effective_dropout_rate)
        var_out = vars_in[0] * (1.0 - effective_dropout_rate)

        return mean_out, var_out


    def _get_output_shape_1d(self, padding, input_shape):
        """
        Calculate the shape of the ZeroPadding1D output
        tensor given the padding size and input shape.
        """
        left_pad, right_pad = padding
        axis_to_pad, features = input_shape
        output_shape = (left_pad + axis_to_pad + right_pad, features)
        return output_shape


    def _get_output_shape_2d(self, padding, input_shape, data_format):
        """
        Calculate the shape of the ZeroPadding2D output
        tensor given the padding size, input shape, and data format.
        """
        ((top_pad, bottom_pad), (left_pad, right_pad)) = padding
        if data_format == CHANNELS_LAST:
            rows, cols, channels = input_shape
            output_shape = (top_pad + rows + bottom_pad,
                            left_pad + cols + right_pad,
                            channels)
        else: # channels_first
            channels, rows, cols = input_shape
            output_shape = (channels,
                            top_pad + rows + bottom_pad,
                            left_pad + cols + right_pad)
        return output_shape


    def _get_output_shape_3d(self, padding, input_shape, data_format):
        """
        Calculate the shape of the ZeroPadding3D output
        tensor given the padding size, input shape, and data format.
        """
        ((left_dim1_pad, right_dim1_pad), \
        (left_dim2_pad, right_dim2_pad), \
        (left_dim3_pad, right_dim3_pad)) = padding
        if data_format == CHANNELS_LAST:
            first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth = input_shape
            output_shape = (left_dim1_pad + first_axis_to_pad + right_dim1_pad,
                            left_dim2_pad + second_axis_to_pad + right_dim2_pad,
                            left_dim3_pad + third_axis_to_pad + right_dim3_pad,
                            depth)
        else: # channels_first
            depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad = input_shape
            output_shape = (depth,
                            left_dim1_pad + first_axis_to_pad + right_dim1_pad,
                            left_dim2_pad + second_axis_to_pad + right_dim2_pad,
                            left_dim3_pad + third_axis_to_pad + right_dim3_pad)
        return output_shape
