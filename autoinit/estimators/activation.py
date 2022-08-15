
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

from numpy import prod

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator

FIXED_ACTIVATION_FNS = {
    'elu': tfkeras.activations.elu,
    'gelu': tfkeras.activations.gelu,
    'linear': tfkeras.activations.linear,
    'relu': tfkeras.activations.relu,
    'selu': tfkeras.activations.selu,
    'sigmoid': tfkeras.activations.sigmoid,
    'softplus': tfkeras.activations.softplus,
    'softsign': tfkeras.activations.softsign,
    'swish': tfkeras.activations.swish,
    'tanh': tfkeras.activations.tanh,
}


class ActivationOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for estimating the output distribution after
    applying a given activation function.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        if isinstance(self.layer, tfkeras.layers.Activation):
            activation_name = self.layer.activation.__name__.lower()
        else:
            activation_name = self.layer.__class__.__name__.lower()

        if activation_name in FIXED_ACTIVATION_FNS.keys():
            activation_fn = FIXED_ACTIVATION_FNS[activation_name]
        elif activation_name == 'leakyrelu':
            alpha = self.layer.alpha
            activation_fn = lambda x : alpha * x if x < 0 else x
        elif activation_name == 'prelu':
            alpha_initializer_name = self.layer.alpha_initializer.__class__.__name__
            if alpha_initializer_name != 'Zeros':
                logging.warning('Alpha initializer %s is not supported yet.  Assuming zero \
alpha initialization for weight init purposes.', alpha_initializer_name)
            activation_fn = FIXED_ACTIVATION_FNS['relu']
        elif activation_name == 'thresholdedrelu':
            theta = self.layer.theta
            activation_fn = lambda x : x if x > theta else 0.0
        elif activation_name == 'softmax':
            # We can't integrate over softmax.  At initialization, we expect
            # balanced logits with mean 1 / NUM_CLASSES and variance 1 / NUM_CLASSES².
            if hasattr(self.layer, 'axis'):
                num_classes = prod(self.layer.input_shape[self.layer.axis])
                # num_classes = prod(self.layer.output.shape[1:])
            else:
                num_classes = prod(self.layer.output.shape[1:])
            mean_out = 1.0 / num_classes
            var_out = 1.0 / math.pow(num_classes, 2)
            # mean_out = 0.0
            # var_out = 1.0 # HACK TODO REMOVE
            # try monte carlo
            import numpy as np
            samples = np.random.normal(loc=means_in[0], scale=np.sqrt(vars_in[0]), size=(int(1e5), num_classes))
            out = np.exp(samples) / np.sum(np.exp(samples), axis=1, keepdims=True)
            mean_out = np.mean(out)
            var_out = np.var(out)

            return mean_out, var_out
        else:
            logging.warning('Activation function %s is not supported.  \
Returning mean and variance unchanged.', activation_name)
            return means_in[0], vars_in[0]

        mean_out, var_out = self._mapped_distribution(activation_fn, means_in[0], vars_in[0])

        return mean_out, var_out
