
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

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from autoinit.components.weighted_sum \
    import WeightedSum
from autoinit.util.replace_layer import ReplaceLayer


class WrapAddition(ReplaceLayer):
    """
    This class inherits from the ReplaceLayer class to define the object
    that will replace each Add layer with a custom WeightedSum layer object.
    """

    def construct_layer_object(self, current_layer_idx: int,
                               current_layer_obj: tfkeras.layers.Layer) -> tfkeras.layers.Layer:
        """
        This function constructs the layer object that needs to replace the layer.
        :param current_layer_obj: Layer object of type layer_type (from constructor) that
        needs to be replaced.
        :return replace_with_obj: The layer that will replace the current layer.
        """
        model_config = self.model.get_config()
        num_inputs = len(model_config['layers'][current_layer_idx]['inbound_nodes'][0])
        replace_with_obj = WeightedSum(num_inputs=num_inputs)

        return replace_with_obj
