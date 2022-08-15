
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

from typing import Dict

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras


class WeightedSum(tfkeras.layers.Layer):
    """
    WeightedSum Layer. This layer scales each input with
    a coefficient before summing them together.
    """

    def __init__(self, coefficients=None, **kwargs):
        """
        The constructor initializes the arguments and the
        base class.
        :param num_inputs: Number of inputs to this Layer
        """
        super().__init__(**kwargs)
        self.coefficients = coefficients
        self.trainable = False

    def build(self, input_shape):
        """
        This function overrides the base function to build the layer.
        :param input_shape: Tuple defining the shape of the input
        """
        pass

    # This call signature is used beginning with TensorFlow 2.5.0.
    def call(self, inputs, *args, **kwargs):
        """
        This function overrides the base function and computes the weighted sum.
        :param inputs:
        :return output: Weighted sum result
        """
        output = tfkeras.layers.add([coeff * inpt for (coeff, inpt) in zip(self.coefficients, inputs)])
        return output

    def get_config(self) -> Dict:
        """
        This function serializes the current layer to construct a configuration.
        :return config: Serialized configuration
        """
        config = super().get_config()
        config.update({
            "coefficients": self.coefficients
        })
        return config
