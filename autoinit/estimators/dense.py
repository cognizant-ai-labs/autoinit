
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

import logging
import math

from typing import List

import numpy as np

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras # pylint: disable=import-error

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator
from autoinit.components.constraints \
    import CenteredUnitNorm

ACTIVATION_FNS = {
    'elu': tfkeras.activations.elu,
    'gelu': tfkeras.activations.gelu,
    'linear': tfkeras.activations.linear,
    'relu': tfkeras.activations.relu,
    'selu': tfkeras.activations.selu,
    'sigmoid': tfkeras.activations.sigmoid,
    'softplus': tfkeras.activations.softplus,
    'softsign': tfkeras.activations.softsign,
    'softmax': tfkeras.activations.softmax,
    'swish': tfkeras.activations.swish,
    'tanh': tfkeras.activations.tanh,
}

DENSE = tfkeras.layers.Dense
CONV_1D = (tfkeras.layers.Conv1D, tfkeras.layers.DepthwiseConv1D)
CONV_2D = (tfkeras.layers.Conv2D, tfkeras.layers.DepthwiseConv2D)
CONV_3D = tfkeras.layers.Conv3D

DENSE_OR_CONV = (tfkeras.layers.Dense,
                 tfkeras.layers.Conv1D,
                 tfkeras.layers.Conv2D,
                 tfkeras.layers.Conv3D)

DEPTHWISE_CONV = (tfkeras.layers.DepthwiseConv1D,
                  tfkeras.layers.DepthwiseConv2D)

DEFAULT_MONTE_CARLO_SAMPLES = 1e5


class DenseOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for initializing the weights and estimating
    the output distribution of Dense and Conv2D layers.
    """

    def _enable_centered_unit_norm(self, gain):
        if isinstance(self.layer, DENSE):
            axis = 0
        elif isinstance(self.layer, CONV_1D):
            axis = [0, 1]
        elif isinstance(self.layer, CONV_2D):
            axis = [0, 1, 2]
        elif isinstance(self.layer, CONV_3D):
            axis = [0, 1, 2, 3]
        constraint = CenteredUnitNorm(axis=axis, gain=gain)
        if isinstance(self.layer, DENSE_OR_CONV):
            self.layer.kernel_constraint = constraint
        elif isinstance(self.layer, DEPTHWISE_CONV):
            self.layer.depthwise_constraint = constraint
        else:
            logging.warning('Unsupported layer type: %s', type(self.layer))

    def estimate(self, means_in: List, vars_in: List):
        """
        Convolution layers can be interpreted as Dense layers with sparse connectivity.  A Dense
        layer can be written as y = Wx+b, where x is the input, W is a weight matrix, b is
        a vector of biases, and y is the result.  Assume the elements of W are mutually
        independent and from the same distribution, and likewise for the elements of x.
        Further assume that W and x are independent of each other.  Then we have
        Var(y) = (# neurons) * Var(Wx).
        Letting W have zero mean and expanding the variance of the product of independent
        variables yields
        Var(y) = (# neurons) [E(W)^2 * Var(x) + Var(W) * E(x)^2 + Var(W) * Var(x)]
               = (# neurons) [Var(W) * E(x)^2 + Var(W) * Var(x)]
               = (# neurons) [Var(W) * (E(x)^2 + Var(x))]
               = (# neurons) [Var(W) * (E(x)^2 + E(x)^2 - E(x)^2)]
               = (# neurons) * Var(W) * E(x^2).
        Initializing so that Var(W) = 1 / [(# neurons) * E(x^2)] will ensure var_out = 1.0.
        Since W and x are assumed to be independent, we also have
        E(y) = E(Wx+b)
             = E(Wx) + E(b)
             = E(Wx)
             = E(W)E(x)
             = 0 * E(x)
             = 0,
        and so we can set mean_out = 0.0.  Although W is sampled from a zero-mean
        distribution, the actual realization may have non-zero empirical mean.  We can force
        it to have zero mean throughout training by re-centering the weights with the
        CenteredUnitNorm constraint.  This constraint also forces the weights to maintain their
        scale throughout training.

        The same analysis applies for Conv1D, Conv2D, Conv3D, and Dense layers.

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        second_moment = vars_in[0] + math.pow(means_in[0], 2)
        scale = self.signal_variance / second_moment
        gain = math.sqrt(scale)
        distribution = self.estimator_config.get("distribution", "truncated_normal")
        if isinstance(self.layer, DEPTHWISE_CONV):
            # For depthwise convolutional layers, each channel is convolved with a different
            # kernel, so we need to multiply the scale/gain by the number of channels.
            scale *= self.layer.input.shape[-1]
            gain *= self.layer.input.shape[-1]
        if distribution == 'orthogonal':
            initializer = tfkeras.initializers.Orthogonal(gain=gain)
        else:
            initializer = tfkeras.initializers.VarianceScaling(scale=scale,
                                                               distribution=distribution)

        if isinstance(self.layer, DENSE_OR_CONV):
            self.layer.kernel_initializer = initializer
        elif isinstance(self.layer, DEPTHWISE_CONV):
            self.layer.depthwise_initializer = initializer
        else:
            logging.warning('Unsupported layer type: %s', type(self.layer))

        if self.estimator_config.get("constrain_weights", False):
            self._enable_centered_unit_norm(gain=gain)

        mean_out = 0.0
        var_out = self.signal_variance

        activation_name = self.layer.activation.__name__
        activation_fn = ACTIVATION_FNS[activation_name]
        if activation_fn is tfkeras.activations.softmax:
            # We can't integrate over softmax, so we use Monte Carlo sampling.
            num_classes = np.prod(self.layer.output.shape[1:])
            num_samples = self.estimator_config.get("monte_carlo_samples",
                                                    DEFAULT_MONTE_CARLO_SAMPLES)
            samples = np.random.normal(loc=means_in[0],
                                       scale=np.sqrt(vars_in[0]),
                                       size=(int(num_samples), num_classes))
            out = np.exp(samples) / np.sum(np.exp(samples), axis=1, keepdims=True)
            mean_out = np.mean(out)
            var_out = np.var(out)
        else:
            mean_out, var_out = self._mapped_distribution(activation_fn, mean_out, var_out)

        return mean_out, var_out
