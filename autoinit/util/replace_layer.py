
# Copyright (C) 2019-2021 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# ENN-release SDK Software in commercial settings.
#
# END COPYRIGHT

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras


class ReplaceLayer:
    """
    This class replaces a given layer type with another.
    """

    def __init__(self, model: tfkeras.Model,
                 layer_type: tfkeras.layers.Layer):
        """
        The constructor initializes the model.
        :param model: TensorFlow/Keras Model. The model must be a flattened model without
        nested Models.  Use the Flattener class to achieve this.
        :param layer_type: The type of the layer that needs to be replaced. Must be
        a class reference.
        """
        self.model = model
        self.layer_type = layer_type

    def construct_layer_object(self, current_layer_idx: int,
                               current_layer_obj: tfkeras.layers.Layer) -> tfkeras.layers.Layer:
        """
        This function constructs the layer object that needs to replace the current layer.
        :param current_layer_idx: Index where the current layer appears in the flat model.
        :param current_layer_obj: Layer object of type layer_type (from constructor) that
        needs to be replaced.
        :return replace_with_obj: The type of layer that will replace the current layer.
        """
        raise NotImplementedError

    def replace(self) -> tfkeras.Model:
        """
        This function replaces the given layer with the custom layer.
        :return replaced_model: Model with the replaced layers
        """
        model_config = self.model.get_config()
        custom_objects = {}
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, self.layer_type):
                layer_config = model_config['layers'][idx]
                replace_with_layer_object = self.construct_layer_object(idx, layer)
                replace_with_layer_class = replace_with_layer_object.__class__
                layer_class_name = replace_with_layer_class.__name__

                if layer_class_name not in custom_objects.keys():
                    custom_objects[layer_class_name] = replace_with_layer_class

                layer_config['class_name'] = layer_class_name
                layer_config['config'] = replace_with_layer_object.get_config()

        replaced_model = tfkeras.Model.from_config(model_config,
                                           custom_objects=custom_objects)
        return replaced_model
