
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

from functools import lru_cache
import logging

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from autoinit.components.constraints \
    import CenteredUnitNorm
from autoinit.initializer.output_distribution_factory \
    import OutputDistributionEstimatorFactory


class AutoInit:
    """
    This class initializes a model's weights to preserve signal variance as much as possible.  Each
    layer receives a mean_in and variance_in from the layer before it.  If the layer has weights,
    those weights are initialized so that variance_out == 1 (in expectation).  Every layer
    (regardless of whether it has weights or not) uses mean_in, variance_in, and the operation
    performed by the layer to compute mean_out and variance_out for consumption by downstream
    layers.  Details specific to each type of layer are provided in each layer's
    OutputDistributionEstimator.  Extending this weight initialization scheme to include other
    types of layers simply requires computing the (expected) output mean and variance of the layer,
    given an input mean and variance.
    """

    def __init__(self,
                 weight_init_config: Dict = None,
                 input_data_mean: float = 0.0,
                 input_data_var: float = 1.0,
                 custom_distribution_estimators: Dict = None):
        """
        The constructor initializes the variables.
        :param weight_init_config: Allows for customizing certain aspects of the weight
            initialization as detailed in the README.
        :param input_data_mean: The mean of the input data.
        :param input_data_var: The variance of the input data.
        :param custom_distribution_estimators: A dictionary mapping custom Layer classes to
            user-defined OutputDistributionEstimators.  This is useful for extending AutoInit
            to new types of layers.
        """
        # If not specified, use an empty dict
        self.weight_init_config = weight_init_config or {}
        self.input_data_mean = input_data_mean
        self.input_data_var = input_data_var
        self.custom_distribution_estimators = custom_distribution_estimators or {}
        self.model = None
        self.mean_var_estimates = {}
        self.estimator_factory = OutputDistributionEstimatorFactory(
            self.weight_init_config, self.custom_distribution_estimators)


    def _get_layer_names(self, nested_list: List[Any]) -> List[str]:
        """
        Sometimes TensorFlow input/output layers or inbound/outbound nodes lists can have
        different shapes or nesting styles.  However, the names of the layers or nodes are always
        in order and are strings, so we just flatten the nested list and return the strings.
        Example:
        [['constant_input_softmod_task_14_node_02', 0, 0],
        ['output_input_softmod_task_14_node_02_block_00', 0, 0],
        ['output_input_softmod_task_14_node_02_block_01', 0, 0]]
        vs.
        [[['constant_input', 0, 0, {}],
        ['model_9_module_2310_1', 15, 0, {}],
        ['maxpool_task_14_node_02_block_01', 0, 0, {}]]]
        """
        array = np.asarray(nested_list).flatten()
        to_return = []
        for thing in array:
            if isinstance(thing, str) and not thing.isdigit():
                to_return.append(thing)
        return to_return


    def _get_config_by_name(self, model: tfkeras.Model, layer_name: str) -> Dict:
        """
        Given a layer name, returns the layer's model config if the layer is
        present at the top level
        :param model: TensorFlow/Keras model
        :param layer_name: Name of the layer whose config is required
        :return layer_config: Dict containing the config of the layer
        """
        config = model.get_config()
        layer_configs = config['layers']
        for layer_config in layer_configs:
            if layer_config['config']['name'] == layer_name:
                return layer_config
        return None


    def _submodel_layer_init(self, layer, submodels):
        """
        This layer is a submodel, so we find its output layer and continue initializing
        """
        submodels = submodels + (layer,) # Go one nested Model deeper
        output_layer_names = self._get_layer_names(layer.get_config()['output_layers'])
        if len(output_layer_names) > 1:
            raise NotImplementedError('Multi-output submodels are not supported yet.')
        return self._initialize_layer(output_layer_names[0], submodels)


    def _outer_input_layer_init(self, layer):
        """
        We are in an InputLayer of the outermost Model; return the input data distribution
        """
        mean_out = self.input_data_mean
        var_out = self.input_data_var
        self.mean_var_estimates[layer.name] = (mean_out, var_out)
        return mean_out, var_out


    def _submodel_input_layer_init(self, layer, submodels):
        """
        We are at a submodel InputLayer (possibly one of multiple)
        """
        submodel = submodels[-1] # Last Model in the list is where we are now
        outer_model = submodels[-2] # And the one before that is this nested Model's outer Model
        submodels = submodels[:-1] # Exit this nested Model and come back to its outer Model

        submodel_config = self._get_config_by_name(outer_model, submodel.name)
        inbound_node_names = self._get_layer_names(submodel_config['inbound_nodes'])
        input_layer_names = self._get_layer_names(submodel.get_config()['input_layers'])

        # Case 1: Just one inbound node
        if len(inbound_node_names) == 1:
            mean_out, var_out = self._initialize_layer(inbound_node_names[0], submodels)

        # Case 2: Just one InputLayer, but multiple inbound nodes.
        # This case might happen in a multi-task learning setup, for example.
        # As long as all inbound nodes share the same distribution there should
        # be no issue.  If this is not the case we just average the statistics
        # as a best guess.  We cannot return a list of means and variances like
        # we would with an Add or Concatenate layer because in this case the
        # downstream layers are expecting only one mean and one variance estimate.
        elif len(input_layer_names) == 1:
            logging.warning('Submodel %s has %s inbound nodes but just one InputLayer. ' \
                'All paths will be initialized, and the average mean and average ' \
                'variance from all inbound nodes will be returned for consumption by ' \
                'downstream Layers.', submodel.name, len(inbound_node_names))

            inbound_data = [self._initialize_layer(name, submodels) for name in inbound_node_names]
            means_in, variances_in = zip(*inbound_data)
            mean_out = sum(means_in) / len(means_in)
            var_out = sum(variances_in) / len(variances_in)

        # Case 3: Multiple inbound nodes and multiple InputLayers
        # Find the inbound node that corresponds to this InputLayer
        elif len(inbound_node_names) == len(input_layer_names):
            idx = input_layer_names.index(layer.name)
            mean_out, var_out = self._initialize_layer(inbound_node_names[idx], submodels)

        # Case 4: Different number of inbound nodes and InputLayers
        else:
            raise ValueError(f'Submodel {submodel.name} has {len(input_layer_names)} ' \
                f'InputLayers but {len(inbound_node_names)} inbound nodes.')

        # Log the mean and variance estimates and return
        self.mean_var_estimates[layer.name] = (mean_out, var_out)
        return mean_out, var_out


    def _standard_layer_init(self, layer, submodels):
        """
        The layer-specific estimator is used to calculate the output statistics
        """
        layer_model = submodels[-1]
        layer_config = self._get_config_by_name(layer_model, layer.name)
        inbound_node_names = self._get_layer_names(layer_config['inbound_nodes'])
        inbound_data = [self._initialize_layer(name, submodels) for name in inbound_node_names]
        means_in, variances_in = zip(*inbound_data)

        estimator = self.estimator_factory.construct_estimation_method(layer)
        mean_out, var_out = estimator.estimate(means_in, variances_in)
        self.mean_var_estimates[layer.name] = (mean_out, var_out)
        return mean_out, var_out


    @lru_cache(maxsize=None) # Memoization
    def _initialize_layer(self,
                         layer_name: str,
                         submodels: Tuple[tfkeras.Model, ...]) -> Tuple[float, float]:
        """
        Recursively initializes layer weights, maintaining unit signal variance when possible.
        :param layer_name: Name of the Layer
        :param submodels: A tuple of Models, beginning with the outermost self.model and ending
            with the Model that this Layer belongs to.  Since TensorFlow allows nested Models,
            this tuple allows for keeping track of where we are in the computation graph.
        :return (mean_out, var_out): Estimates of the output statistics of this Layer
        """
        layer_model = submodels[-1]
        layer = layer_model.get_layer(layer_name)

        if isinstance(layer, tfkeras.Model):
            return self._submodel_layer_init(layer, submodels)

        if isinstance(layer, tfkeras.layers.InputLayer):
            if layer_model is self.model:
                return self._outer_input_layer_init(layer)
            return self._submodel_input_layer_init(layer, submodels)

        return self._standard_layer_init(layer, submodels)


    def initialize_model(self,
                         model: Union[tfkeras.Model, Dict, str],
                         return_estimates: bool = False,
                         custom_objects: Dict = None):
        """
        This function is the entry point to this class.
        :param model: The TensorFlow Model to be initialized or a dictionary config or
            JSON string defining the model
        :param return_estimates: Whether to return a Dict of layer mean and variance estimates
        :param custom_objects: A dictionary of custom objects such as Layers, Constraints,
            and so on, needed for the Model to be JSON-serializable.
        :return self.model: Model with weights initialized
        """
        self.mean_var_estimates = {}
        custom_objects = custom_objects or {}

        if isinstance(model, tfkeras.Model):
            self.model = model
        elif isinstance(model, dict):
            self.model = tfkeras.Model.from_config(model, custom_objects=custom_objects)
        elif isinstance(model, str):
            self.model = tfkeras.models.model_from_json(model, custom_objects=custom_objects)
        else:
            raise TypeError('The model argument must be a TensorFlow Model, config dictionary, '\
                           f'or JSON string.  Got type {type(model)}.')

        if len(self.model.inputs) != 1 or len(self.model.outputs) != 1:
            logging.warning("Weight initialization for multi input/output models is experimental!")

        # Sequential model configs don't have an 'output_layers' key,
        # so they are converted to Functional models instead.
        if isinstance(self.model, tfkeras.Sequential):
            self.model = tfkeras.Model(self.model.inputs, self.model.outputs)

        # Since the initialization is deterministic and the model is a DAG, we can
        # recursively initialize from all output layers, and then the whole model
        # should be propertly initialized.
        for output_layer in self.model.get_config()['output_layers']:
            output_layer_name = output_layer[0]
            self._initialize_layer(output_layer_name, (self.model,))

        # The model must be reinstantiated for the weight initialization to take effect
        custom_objects.update({'CenteredUnitNorm': CenteredUnitNorm})
        self.model = tfkeras.Model.from_config(self.model.get_config(),
                                       custom_objects=custom_objects)

        if return_estimates:
            return self.model, self.mean_var_estimates
        return self.model
