
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

from numpy import average, prod

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras # pylint: disable=import-error

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class TFOpLambdaOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    TensorFlow wraps API calls (like tf.nn.relu) in a TFOpLambda Layer object.
    Similarly, sometimes users wrap arbitrary functions in Lambda layers.
    In these cases, the best we can do is to try to infer the layer operation
    from the layer name.  Often the layer names will have suffixes, like tf.reshape_3,
    so we just check whether these API calls are in the layer name.  This class is a
    workaround.  If possible, API calls should be replaced with proper Layer objects.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Map layer names to estimators.
        self.tf_op_lambda_estimators = {
            'tf.compat.v1.transpose': self._pass_through,
            'tf.concat' : self._concat,
            'tf.image.resize' : self._pass_through,
            'tf.math.multiply' : self._multiply,
            'tf.math.reduce_mean' : self._reduce_mean,
            'tf.math.reduce_sum' : self._reduce_sum,
            'tf.matmul' : self._matmul,
            'tf.nn.relu' : self._relu,
            'tf.reshape': self._pass_through,
            'tf.split' : self._pass_through,
            'tf.transpose': self._pass_through,
        }

    def _pass_through(self, means_in, vars_in):
        # These layers don't alter the input distribution.
        return means_in[0], vars_in[0]

    def _concat(self, means_in, vars_in):
        # See concatenate.py
        sizes = [prod(inpt.shape[1:]) for inpt in self.layer.input]
        mean_out = average(means_in, weights=sizes)
        second_moments = [
            variance + math.pow(mean, 2) for mean, variance in zip(means_in, vars_in)]
        second_moment = average(second_moments, weights=sizes)
        var_out = second_moment - math.pow(mean_out, 2)
        return mean_out, var_out

    def _multiply(self, means_in, vars_in):
        # See multiply.py
        mean_out = prod(means_in)
        var_out = prod([var + math.pow(mean, 2) for mean, var in zip(means_in, vars_in)]) - \
                  prod([math.pow(mean, 2) for mean in means_in])
        return mean_out, var_out

    def _reduce_mean(self, means_in, vars_in):
        # See add.py and multiply.py
        # Infer the dimensions being averaged across by the input and output shapes.
        # Assumes keepdims=True.
        # For example, if the input shape is (4, 100, 64) and the output shape is (4, 1, 64),
        # then we averaged 100 entries along the second axis.
        dimensions_averaged = 1
        for dim in zip(self.layer.input.shape, self.layer.output.shape):
            if dim[0] != dim[1]:
                dimensions_averaged *= max(dim)
        mean_out = means_in[0]
        var_out = vars_in[0] / dimensions_averaged
        return mean_out, var_out

    def _reduce_sum(self, means_in, vars_in):
        # See add.py
        # Infer the dimensions being summed across by the input and output shapes.
        # Assumes keepdims=True.
        # For example, if the input shape is (4, 100, 64) and the output shape is (4, 1, 64),
        # then we summed 100 entries along the second axis.
        dimensions_summed = 1
        for dim in zip(self.layer.input.shape, self.layer.output.shape):
            if dim[0] != dim[1]:
                dimensions_summed *= max(dim)
        mean_out = dimensions_summed * means_in[0]
        var_out = dimensions_summed * vars_in[0]
        return mean_out, var_out

    def _matmul(self, means_in, vars_in):
        # dimension [..., m, n] * [..., n, p] -> [..., m, p]
        # This calculation assumes that transpose_a, transpose_b, adjoint_a, and adjoint_b
        # are all False in the call to tf.matmul and that the inputs are independent.
        # Each entry is the sum of the products of the corresponding entries of the inputs:
        # output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j

        common_dimension = self.layer.input[0].shape[-1]
        mean_out = common_dimension * means_in[0] * means_in[1]
        prod_var =  prod(
            [var + math.pow(mean, 2) for mean, var in zip(means_in, vars_in)]) - \
            prod([math.pow(mean, 2) for mean in means_in])
        var_out = common_dimension * prod_var

        if 'tf.matmul_x_' in self.layer.name:
            # Special case where the matrix product is scaled by a constant.
            constant = float(self.layer.name.split('_x_')[1])
            mean_out *= constant
            var_out *= constant ** 2

        if common_dimension != self.layer.input[1].shape[-2]:
            logging.warning('tf.matmul layer %s has mismatched dimensions. ' \
                'If either matrix needs to be transposed, do so outside of the ' \
                'call to tf.matmul. Returning mean and variance unchanged.', self.layer.name)
            mean_out = means_in[0]
            var_out = vars_in[0]

        return mean_out, var_out

    def _relu(self, means_in, vars_in):
        # See activation.py
        return self._mapped_distribution(tfkeras.activations.relu, means_in[0], vars_in[0])

    def estimate(self, means_in: List, vars_in: List):
        """
        :param means_in: List containing the means of the input distributions of
        the incoming layers
        :param vars_in: List containing the variances of the input distributions of
        the incoming layers

        :return mean_out: Mean of the output distribution
        :return var_out: Variance of the output distribution
        """

        logging.warning('AutoInit will try to infer the behavior of layer ' \
            '%s from its name.  If possible, replace API calls like ' \
            'tf.math.multiply with Layer objects like tf.keras.layers.Multiply.',
            self.layer.name)

        # Map the layer name to the correct estimator.
        supported_layer_names = self.tf_op_lambda_estimators.keys()
        for layer_name in supported_layer_names:
            if layer_name in self.layer.name.lower():
                return self.tf_op_lambda_estimators[layer_name](means_in, vars_in)

        # If we get here, we couldn't infer the behavior of the layer.
        logging.warning('Layer %s of type %s is not supported.  ' \
            'Returning mean and variance unchanged.',
            self.layer.name, self.layer.__class__.__name__)
        mean_out = means_in[0]
        var_out = vars_in[0]

        return mean_out, var_out
