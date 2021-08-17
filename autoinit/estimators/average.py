
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

from framework.enn.autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class AverageOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of Average layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        By linearity of expectation:
        E[(X1 + X2 + ... + XN) / N] = E[X1 / N] + E[X2 / N] + ... + E[XN / N]
                                    = (E[X1] + E[X2] + ... + E[XN]) / N.

        If we assume the inputs are independent, we also have:
        Var((X1 + X2 + ... + XN) / N) = Var(X1 / N) + Var(X2 / N) + ... + Var(XN / N)
                                      = (Var(X1) + Var(X2) + ... + Var(XN)) / N².

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        mean_out = sum(means_in) / len(means_in)
        var_out = sum(vars_in) / math.pow(len(vars_in), 2)

        return mean_out, var_out
