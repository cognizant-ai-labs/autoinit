
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

from typing import Dict
from typing import List

from numpy import inf

from scipy import integrate
from scipy.stats import norm

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras


class LayerOutputDistributionEstimator:
    """
    An interface describing the contract that every layer type(s)
    must implement to estimate the output distribution.
    """

    def __init__(self, layer: tfkeras.layers.Layer, estimator_config: Dict):
        """
        This constructor initializes the params.
        :param layer: TensorFlow/Keras Layer
        :param estimator_config: Dictionary containing the required config for
        layer.
        """
        self.layer = layer
        self.estimator_config = estimator_config

    def estimate(self, means_in: List, vars_in: List):
        """
        This function must be overridden by the layer to compute the mean_out
        and the var_out.
        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """
        raise NotImplementedError

    def _mapped_distribution(self,
                             function,
                             input_mean: float = 0,
                             input_var: float = 1) -> (float, float):
        """
        Given input distributed as N(input_mean, sqrt(input_var)), calculates the mean and
        variance of the resulting distribution after applying the function 'function'.  The
        distribution is calculated using the law of the unconscious statistician.  See
        https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician.
        :param function: Function to apply
        :param input_mean: Mean of input distribution
        :param input_var: Variance of input distribution
        :return new_mean: Mean of new distribution
        :return new_var: Variance of new distribution.
        """
        gaussian = norm(loc=input_mean, scale=math.sqrt(input_var))
        new_mean = integrate.quad(lambda x: function(x) * gaussian.pdf(x), -inf, inf)[0]
        second_moment = integrate.quad(
            lambda x: math.pow(function(x), 2) * gaussian.pdf(x), -inf, inf)[0]
        new_var = second_moment - math.pow(new_mean, 2)
        return new_mean, new_var
