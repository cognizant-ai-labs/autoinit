
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

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class BatchNormalizationOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of BatchNormalization
    layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        BatchNormalization normalizes the input to have zero mean and unit variance.
        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        mean_out = 0.0
        var_out = 1.0

        return mean_out, var_out
