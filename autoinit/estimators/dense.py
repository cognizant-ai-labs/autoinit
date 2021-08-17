
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

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

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


class DenseOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for initializing the weights and estimating
    the output distribution of Dense and Conv2D layers.
    """

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
        scale = 1.0 / second_moment
        gain = math.sqrt(scale)
        distribution = self.estimator_config.get("distribution",
                                                 "truncated_normal")
        if distribution == 'orthogonal':
            self.layer.kernel_initializer = tfkeras.initializers.Orthogonal(gain=gain)
        else:
            self.layer.kernel_initializer = tfkeras.initializers.VarianceScaling(scale=scale,
                                                            distribution=distribution)
        if self.estimator_config.get("constrain_weights", False):
            if isinstance(self.layer, tfkeras.layers.Dense):
                axis = 0
            elif isinstance(self.layer, tfkeras.layers.Conv1D):
                axis = [0, 1]
            elif isinstance(self.layer, tfkeras.layers.Conv2D):
                axis = [0, 1, 2]
            elif isinstance(self.layer, tfkeras.layers.Conv3D):
                axis = [0, 1, 2, 3]
            self.layer.kernel_constraint = CenteredUnitNorm(axis=axis, gain=gain)
        mean_out = 0.0
        var_out = 1.0

        activation_name = self.layer.activation.__name__
        activation_fn = ACTIVATION_FNS[activation_name]
        if activation_fn is tfkeras.activations.softmax:
            # We can't integrate over softmax.  At initialization, we expect
            # balanced logits with mean 1 / NUM_CLASSES and variance 1 / NUM_CLASSES².
            num_classes = prod(self.layer.output.shape[1:])
            mean_out = 1.0 / num_classes
            var_out = 1.0 / math.pow(num_classes, 2)
        else:
            mean_out, var_out = self._mapped_distribution(activation_fn, mean_out, var_out)

        return mean_out, var_out
