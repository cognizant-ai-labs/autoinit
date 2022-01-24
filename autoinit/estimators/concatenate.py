
# Copyright (C) 2021-2022 Cognizant Digital Business, Evolutionary AI.
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

from numpy import average, prod

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class ConcatenateOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of concatenate
    layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        Given the concatenated distribution [X1, X2, ..., XN] where distribution
        Xi has Ci elements, the expected mean is
        mean_out = (C1 * E[X1] + C2 * E[X2] + ... + CN * E[XN]) / (C1 + C2 + ... + CN).

        Similarly, the second moment of the concatenated distribution is
        second_moment = (C1 * E[X1²] + C2 * E[X2²] + ... + CN * E[XN²]) / (C1 + C2 + ... + CN).

        The outgoing variance can then be expressed as
        second_moment - mean_out².

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        # Calculate the size of each input, ignoring the batch dimension
        sizes = [prod(inpt.shape[1:]) for inpt in self.layer.input]

        mean_out = average(means_in, weights=sizes)
        second_moments = [variance + math.pow(mean, 2) for mean, variance in zip(means_in, vars_in)]
        second_moment = average(second_moments, weights=sizes)
        var_out = second_moment - math.pow(mean_out, 2)

        return mean_out, var_out
