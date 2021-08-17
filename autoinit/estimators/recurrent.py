
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
import logging

from typing import List

import numpy as np

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator

DEFAULT_MONTE_CARLO_SAMPLES = 1e4

class RecurrentOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    This class is responsible for initializing the weights and estimating
    the output distribution of SimpleRNN, GRU, and LSTM layers.
    """

    def estimate(self, means_in: List, vars_in: List):
        """
        For simplicity, this estimator uses Monte Carlo simulation to calculate
        mean and variance statistics for SimpleRNN, GRU, and LSTM layers.  Although
        these layers have configurable weights, they are intentionally left unchanged.
        The reason for this setup is as follows.

        A GRU cell takes a previous hidden state h_tm1 and input x_t and
        produces the next hidden state h_t.  The GRU has a kernel W,
        recurrent kernel U, and bias b.  The equations which define a GRU are:

        z = sigmoid(W_z * x_t + U_z * h_tm1 + b_z)
        r = sigmoid(W_r * x_t + U_r * h_tm1 + b_r)
        h' = tanh(W_h * x_t + r * U_h * h_tm1 + b_h)
        h_t = z * h_tm1 + (1 - z) * h'

        The output vector of a GRU has entries bounded between -1 and 1.  This is
        because the range of sigmoid is (0, 1) while the range of tanh is (-1, 1).
        Thus, every step computes a weighted average (depending on z) of the previous
        hidden state h_tm1 and h', which is between -1 and 1.

        Because of this structure, it doesn't make sense to try to scale the GRU's
        weights to control the output variance.  Indeed, the only case in which the
        GRU's output variance is one is the degenerate case where all values in h_t
        are either -1 or 1.

        Instead, we perform a Monte Carlo simulation by creating a copy of the GRU
        and passing random noise through it sampled from a normal distribution
        with the incoming mean and variance.  Outgoing mean and variance are calculated
        based on the results of this simulation.  We use the same approach for
        SimpleRNN and LSTM.

        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """
        # Generate samples
        num_samples = int(self.estimator_config.get("recurrent_monte_carlo_samples",
                                                    DEFAULT_MONTE_CARLO_SAMPLES))
        size = [num_samples] + list(self.layer.input_shape[1:])
        samples = np.random.normal(loc=means_in[0], scale=np.sqrt(vars_in[0]), size=size)

        # Create copy of the layer
        if isinstance(self.layer, tfkeras.layers.SimpleRNN):
            dummy_layer = tfkeras.layers.SimpleRNN.from_config(self.layer.get_config())
        elif isinstance(self.layer, tfkeras.layers.GRU):
            dummy_layer = tfkeras.layers.GRU.from_config(self.layer.get_config())
        elif isinstance(self.layer, tfkeras.layers.LSTM):
            dummy_layer = tfkeras.layers.LSTM.from_config(self.layer.get_config())
        else:
            raise NotImplementedError(f'Layer {type(self.layer).__name__} is not supported yet')

        if dummy_layer.return_sequences or dummy_layer.return_state:
            logging.warning('Layer %s is returning its output sequence and/or final state ' \
                'in addition to its regular output.  These values will be accounted for ' \
                'in mean and variance estimation.  If the values are not actually consumed by ' \
                'downstream layers, this could make output distribution estimation less accurate.',
                self.layer.name)

        # Pass samples through layer and calculate statistics
        output = dummy_layer(samples)
        mean_out = np.mean(output)
        var_out = np.var(output)

        return mean_out, var_out
