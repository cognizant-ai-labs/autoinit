
# Copyright (C) 2019-2021 Cognizant Digital Business, Evolutionary AI.
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

from framework.enn.autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class WeightedSumOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of WeightedSum
    layers, in addition to calculating optimal coefficients for the layer.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        Given inputs X1, X2, ..., XN with variances var1, var2, ... varN, and coefficients
        a1, a2, ... aN, if we assume zero covariance (which may not be the case), we have
        Var(a1X1 + a2X2 + ... aNXN) = a1^2 * Var(X1) + a2^2 * Var(X2) + ... + aN^2 * Var(XN).
        Setting aI = 1.0 / sqrt(Var(XI) * N) is sufficient to ensure var_out = 1.0.  Under
        this setting, we have
        mean_out = a1 * E(X1) + a2 * E(X2) + ... aN * E(XN),
        assuming all inputs have the same size.

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        num_inputs = len(vars_in)
        weights = []
        for variance_in in vars_in:
            weights.append(1.0 / math.sqrt(variance_in * num_inputs))
        mean_out = sum([weight * mean for weight, mean in zip(weights, means_in)])
        weights = np.asarray(weights).reshape(-1, 1)
        self.layer.set_weights(weights)
        var_out = 1.0

        return mean_out, var_out
