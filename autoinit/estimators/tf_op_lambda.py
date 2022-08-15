
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
import tensorflow.keras as tfkeras

from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator


class TFOpLambdaOutputDistributionEstimator(LayerOutputDistributionEstimator):
    """
    TensorFlow wraps API calls (like tf.nn.relu) in a TFOpLambda Layer object.
    Similarly, sometimes users wrap arbitrary functions in Lambda layers.
    In these cases, the best we can do is to try to infer the layer operation 
    from the layer name.  Often the layer names will have suffixes, like tf.reshape_3,
    so we just check whether these API calls are in the layer name.
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

        logging.info('AutoInit will try to infer the behavior of layer ' \
            f'{self.layer.name} from its name.  If possible, replace API ' \
            'calls like tf.math.multiply with Layer objects like tf.keras.layers.Multiply.')

        layer_name = self.layer.name.lower()
        
        # These layers don't alter the input distribution.
        if  'tf.compat.v1.transpose' in layer_name or \
            'tf.transpose'           in layer_name or \
            'tf.image.resize'        in layer_name or \
            'tf.reshape'             in layer_name or \
            'tf.split'               in layer_name:
            mean_out = means_in[0]
            var_out = vars_in[0]

        # See concatenate.py
        elif 'tf.concat' in layer_name:
            sizes = [prod(inpt.shape[1:]) for inpt in self.layer.input]
            mean_out = average(means_in, weights=sizes)
            second_moments = [variance + math.pow(mean, 2) for mean, variance in zip(means_in, vars_in)]
            second_moment = average(second_moments, weights=sizes)
            var_out = second_moment - math.pow(mean_out, 2)

        # See multiply.py
        elif 'tf.math.multiply' in layer_name:
            mean_out = prod(means_in)
            var_out = prod([var + math.pow(mean, 2) for mean, var in zip(means_in, vars_in)]) - \
                  prod([math.pow(mean, 2) for mean in means_in])

        # See add.py
        elif 'tf.math.reduce_mean' in layer_name or 'tf.math.reduce_sum' in layer_name:
            # Infer the dimensions being summed across by the input and output shapes.
            # Assumes keepdims=True.
            # For example, if the input shape is (4, 100, 64) and the output shape is (4, 1, 64),
            # then we summed 100 entries along the second axis.
            input_shape = self.layer.input.shape
            output_shape = self.layer.output.shape
            dimensions_summed = 1
            for dim in zip(input_shape, output_shape):
                if dim[0] != dim[1]:
                    dimensions_summed *= max(dim)
            # mean_out = dimensions_summed * means_in[0]
            # var_out = dimensions_summed * vars_in[0]
            if 'tf.math.reduce_mean' in layer_name:
                mean_out = means_in[0]
                var_out = vars_in[0] / dimensions_summed
            elif 'tf.math.reduce_sum' in layer_name:
                mean_out = dimensions_summed ** 1 * means_in[0] # TODO test this
                var_out = dimensions_summed ** 1.5 * vars_in[0] # actually correct? why? --> compromise between no correlation **1 and perfect correlation **2, seems like **1 is theone we should do / is the one in the paper
            # mean_out = means_in[0] # HACK TODO REMOVE
            # var_out = vars_in[0] # HACK TODO REMOVE
            # mean_out = means_in[0] / dimensions_summed
            # var_out = vars_in[0] / dimensions_summed

        elif 'tf.matmul' in layer_name:
            # dimension [..., m, n] * [..., n, p] -> [..., m, p]
            # This calculation assumes that transpose_a, transpose_b, adjoint_a, and adjoint_b 
            # are all False in the call to tf.matmul.
            common_dimension = self.layer.input[0].shape[-1]
            if common_dimension != self.layer.input[1].shape[-2]:
                logging.warning('tf.matmul layer %s has mismatched dimensions. ' \
                    'If either matrix needs to be transposed, do so outside of the ' \
                    'call to tf.matmul. Returning mean and variance unchanged.', self.layer.name)
                mean_out = means_in[0]
                var_out = vars_in[0]

            else:
                # Assume independent inputs.
                # Each entry is the sum of the products of the corresponding entries of the two inputs:
                # output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j
                mean_out = common_dimension * means_in[0] * means_in[1]
                prod_var =  prod([var + math.pow(mean, 2) for mean, var in zip(means_in, vars_in)]) - \
                    prod([math.pow(mean, 2) for mean in means_in])
                var_out = common_dimension * prod_var

            if 'tf.matmul_x_' in layer_name:
                # Special case where the matrix product is scaled by a constant.
                constant = float(layer_name.split('_x_')[1])
                mean_out *= constant
                var_out *= constant ** 2

            else:
                # TODO HACK REMOVE
                var_out = 1.0


        # See activation.py
        elif 'tf.nn.relu' in layer_name:
            mean_out, var_out = self._mapped_distribution(
                tfkeras.activations.relu, means_in[0], vars_in[0])

        else:
            logging.warning('Layer %s of type %s is not supported.  ' \
                'Returning mean and variance unchanged.',
                self.layer.name, self.layer.__class__.__name__)
            mean_out = means_in[0]
            var_out = vars_in[0]

        return mean_out, var_out
