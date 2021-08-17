
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

from numpy import prod

from framework.enn.autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class MultiplyOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of Multiply layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        Assuming the inputs are independent:
        E[X1 * X2 * ... * XN] = E[X1] * E[X2] * ... * E[XN].

        This assumption also allows us to derive:
        Var(X1 * X2 * ... * XN) = E[(X1 * X2 * ... * XN)²] - (E[X1 * X2 * ... * XN])²
                                = E[X1² * X2² * ... * XN²] - (E[X1] * E[X2] * ... * E[XN])²
                                = E[X1²] * E[X2²] * ... * E[XN²] - E[X1]² * E[X2]² * ... * E[XN]²
                                = Π (Var(Xi) + E[Xi]²) - Π E[Xi]².

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        mean_out = prod(means_in)
        var_out = prod([var + math.pow(mean, 2) for mean, var in zip(means_in, vars_in)]) - \
                  prod([math.pow(mean, 2) for mean in means_in])

        return mean_out, var_out
