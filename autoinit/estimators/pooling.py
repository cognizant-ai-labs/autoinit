
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
import math
from typing import List

import numpy as np

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator

CHANNELS_LAST = 'channels_last'
DEFAULT_GLOBAL_POOLING_SAMPLES = 1e4
DEFAULT_MONTE_CARLO_SAMPLES = 1e7


class PoolingOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of pooling
    layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        Pooling layers apply a function across sliding window of size k x k.  For example,
        AveragePooling2D computes the average of all values in the window, while MaxPooling2D
        takes the maximum.  Assume the input has independent pixels normally distributed with
        mean μ and standard deviation σ.  If the layer is AveragePooling2D, let f be the average
        function.  Similarly, if the layer is MaxPooling2D, let f be the maximum function.  Let
        PDF(μ, σ, x) denote the probability density function of a N(μ, σ) random variable evaluated
        at x.  We have:

        mean_out = ∫∫...∫ f(x_1, x_2, ..., x_k²) *
                        PDF(μ, σ, x_1) * PDF(μ, σ, x_2) * ... * PDF(μ, σ, x_k²) dx_1 dx_2 ... dx_k²,
        second_moment = ∫∫...∫ [f(x_1, x_2, ..., x_k²)]² *
                        PDF(μ, σ, x_1) * PDF(μ, σ, x_2) * ... * PDF(μ, σ, x_k²) dx_1 dx_2 ... dx_k²,
        var_out = second_moment - mean_out².

        Unfortunately, a typical 3 x 3 pooling layer requires computing 9 nested integrals which is
        prohibitively expensive.  A more efficient alternative is Monte Carlo integration.

        Sample xᵢ_1, xᵢ_2, ..., xᵢ_k² ∼ N(μ, σ) for i = 1, ..., NUM_SAMPLES.
        mean_out ≈ ∑ f(xᵢ_1, xᵢ_2, ..., xᵢ_k²) / NUM_SAMPLES,
        second_moment ≈ ∑ [f(xᵢ_1, xᵢ_2, ..., xᵢ_k²)]² / NUM_SAMPLES,
        var_out ≈ second_moment - mean_out².

        The estimation quality improves as NUM_SAMPLES is increased.

        The same analysis applies to 1D or 3D pooling layers, except the sliding window will be of
        size k in the 1D case and size k x k x k in the 3D case.

        In the case of GlobalAveragePooling or GlobalMaxPooling layers, the window is the size
        of the input across all axes except the batch and channel dimensions.

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """
        operation = self._get_operation()
        sample_size = self._get_sample_size()
        samples = np.random.normal(loc=means_in[0],
                                   scale=math.sqrt(vars_in[0]),
                                   size=sample_size)
        mean_out = np.mean(operation(samples, axis=0))
        var_out = np.var(operation(samples, axis=0))

        return mean_out, var_out


    def _get_operation(self):
        """
        Returns a mean or maximum operator, depending on the Layer type
        """
        # AveragePooling Layers
        if isinstance(self.layer, (tfkeras.layers.AveragePooling1D,
                                   tfkeras.layers.AveragePooling2D,
                                   tfkeras.layers.AveragePooling3D,
                                   tfkeras.layers.GlobalAveragePooling1D,
                                   tfkeras.layers.GlobalAveragePooling2D,
                                   tfkeras.layers.GlobalAveragePooling3D)):
            operation = np.mean

        # MaxPooling Layers
        elif isinstance(self.layer, (tfkeras.layers.MaxPooling1D,
                                     tfkeras.layers.MaxPooling2D,
                                     tfkeras.layers.MaxPooling3D,
                                     tfkeras.layers.GlobalMaxPooling1D,
                                     tfkeras.layers.GlobalMaxPooling2D,
                                     tfkeras.layers.GlobalMaxPooling3D)):
            operation = np.amax

        else:
            raise NotImplementedError(f'Layer {type(self.layer).__name__} is not supported yet')

        return operation


    def _get_sample_size(self):
        """
        Returns the sample size used for Monte Carlo sampling, depending on the Layer type
        """
        # Pooling Layers with sliding windows
        if isinstance(self.layer, (tfkeras.layers.AveragePooling1D,
                                   tfkeras.layers.AveragePooling2D,
                                   tfkeras.layers.AveragePooling3D,
                                   tfkeras.layers.MaxPooling1D,
                                   tfkeras.layers.MaxPooling2D,
                                   tfkeras.layers.MaxPooling3D)):
            num_samples = self.estimator_config.get("monte_carlo_samples",
                                                    DEFAULT_MONTE_CARLO_SAMPLES)
            dimension = np.prod(self.layer.pool_size)

        # Global Pooling Layers
        elif isinstance(self.layer, (tfkeras.layers.GlobalAveragePooling1D,
                                     tfkeras.layers.GlobalAveragePooling2D,
                                     tfkeras.layers.GlobalAveragePooling3D,
                                     tfkeras.layers.GlobalMaxPooling1D,
                                     tfkeras.layers.GlobalMaxPooling2D,
                                     tfkeras.layers.GlobalMaxPooling3D)):
            num_samples = self.estimator_config.get("global_pooling_samples",
                                                    DEFAULT_GLOBAL_POOLING_SAMPLES)
            if self.layer.data_format == CHANNELS_LAST:
                # ignore batch (first) and channels (last) axes
                axes_sizes = self.layer.input_shape[1:-1]
            else: # channels_first
                # ignore the batch (first) and channels (second) axes
                axes_sizes = self.layer.input_shape[2:]
            dimension = np.prod(axes_sizes)

        else:
            raise NotImplementedError(f'Layer {type(self.layer).__name__} is not supported yet')

        return (dimension, int(num_samples))
