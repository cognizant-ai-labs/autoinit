
# Copyright (C) 2019-2021 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# ENN-release SDK Software in commercial settings.
#
# END COPYRIGHT
from typing import List

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from framework.enn.autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class DropoutOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of dropout
    layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        Dropout randomly sets drop_rate neurons to zero.
        mean_in = sum(kept_neurons) / (# kept_neurons)
        mean_out = sum(kept_neurons) / [(# kept_neurons) + (# dropped_neurons)]
                 = mean_in * [(# kept_neurons) / (# neurons)]
                 = mean_in * (1.0 - drop_rate)
        Recall: Var(X) = E(X^2) - E(X)^2
        var_out = second_moment_out - mean_out^2
                = second_moment_out - [mean_in * (1.0 - drop_rate)]^2
        Notice that a sample from X^2 = 0 if and only if it was 0 originally or it was
        dropped.  Therefore the same analysis applies to the random variable X^2 and we can
        write:
        var_out = second_moment_in * (1.0 - drop_rate) - [mean_in * (1.0 - drop_rate)]^2
                = var_in * (1.0 - drop_rate)

        The above analysis applies for SpatialDropout1D, SpatialDropout2D, and SpatialDropout3D
        layers.  For Dropout layers, TensorFlow automatically scales the remaining neurons by
        1.0 / (1.0 - drop_rate).  Therefore, in this case we have:
        mean_out = mean_in * (1.0 - drop_rate) / (1.0 - drop_rate)
                 = mean_in
        and
        var_out = var_in * (1.0 - drop_rate) / (1.0 - drop_rate)^2
                = var_in / (1.0 - drop_rate).

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        drop_rate = self.layer.get_config()['rate']
        if isinstance(self.layer, tfkeras.layers.Dropout):
            mean_out = means_in[0]
            var_out = vars_in[0] / (1.0 - drop_rate)
        elif isinstance(self.layer, (tfkeras.layers.SpatialDropout1D,
                                     tfkeras.layers.SpatialDropout2D,
                                     tfkeras.layers.SpatialDropout3D)):
            mean_out = means_in[0] * (1.0 - drop_rate)
            var_out = vars_in[0] * (1.0 - drop_rate)

        return mean_out, var_out
