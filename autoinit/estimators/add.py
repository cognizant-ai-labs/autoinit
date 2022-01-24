
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
from typing import List

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class AddOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution of Add layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        By linearity of expectation:
        E[X1 + X2 + ... + XN] = E[X1] + E[X2] + ... + E[XN].

        If we assume the inputs are independent, we also have:
        Var(X1 + X2 + ... + XN) = Var(X1) + Var(X2) + ... + Var(XN).

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        mean_out = sum(means_in)
        var_out = sum(vars_in)

        return mean_out, var_out
