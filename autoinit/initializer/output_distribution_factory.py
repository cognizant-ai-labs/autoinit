
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

from typing import Dict

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from autoinit.components.weighted_sum \
    import WeightedSum

from autoinit.estimators.activation \
    import ActivationOutputDistributionEstimator
from autoinit.estimators.add \
    import AddOutputDistributionEstimator
from autoinit.estimators.average \
    import AverageOutputDistributionEstimator
from autoinit.estimators.batch_norm \
    import BatchNormalizationOutputDistributionEstimator
from autoinit.estimators.concatenate \
    import ConcatenateOutputDistributionEstimator
from autoinit.estimators.dense \
    import DenseOutputDistributionEstimator
from autoinit.estimators.dropout \
    import DropoutOutputDistributionEstimator
from autoinit.estimators.estimate_layer_output_distribution \
    import LayerOutputDistributionEstimator
from autoinit.estimators.multiply \
    import MultiplyOutputDistributionEstimator
from autoinit.estimators.passthrough \
    import PassThroughOutputDistributionEstimator
from autoinit.estimators.pooling \
    import PoolingOutputDistributionEstimator
from autoinit.estimators.recurrent \
    import RecurrentOutputDistributionEstimator
from autoinit.estimators.subtract \
    import SubtractOutputDistributionEstimator
from autoinit.estimators.weighted_sum \
    import WeightedSumOutputDistributionEstimator
from autoinit.estimators.zero_padding \
    import ZeroPaddingOutputDistributionEstimator


class OutputDistributionEstimatorFactory:
    """
    This class holds the supported layer types and constructs the
    appropriate class to compute the output distribution estimates.
    """

    def __init__(self, estimator_config: Dict, custom_distribution_estimators: Dict = None):
        """
        The constructor consists of the supported layer types.
        """
        custom_distribution_estimators = custom_distribution_estimators or {}
        self.builtin_layers = {
            tfkeras.layers.Activation             : ActivationOutputDistributionEstimator,
            tfkeras.layers.Add                    : AddOutputDistributionEstimator,
            tfkeras.layers.Average                : AverageOutputDistributionEstimator,
            tfkeras.layers.AveragePooling1D       : PoolingOutputDistributionEstimator,
            tfkeras.layers.AveragePooling2D       : PoolingOutputDistributionEstimator,
            tfkeras.layers.AveragePooling3D       : PoolingOutputDistributionEstimator,
            tfkeras.layers.BatchNormalization     : BatchNormalizationOutputDistributionEstimator,
            tfkeras.layers.Cropping1D             : PassThroughOutputDistributionEstimator,
            tfkeras.layers.Cropping2D             : PassThroughOutputDistributionEstimator,
            tfkeras.layers.Cropping3D             : PassThroughOutputDistributionEstimator,
            tfkeras.layers.Concatenate            : ConcatenateOutputDistributionEstimator,
            tfkeras.layers.Conv1D                 : DenseOutputDistributionEstimator,
            tfkeras.layers.Conv2D                 : DenseOutputDistributionEstimator,
            tfkeras.layers.Conv3D                 : DenseOutputDistributionEstimator,
            tfkeras.layers.Dense                  : DenseOutputDistributionEstimator,
            tfkeras.layers.Dropout                : DropoutOutputDistributionEstimator,
            tfkeras.layers.ELU                    : ActivationOutputDistributionEstimator,
            tfkeras.layers.Flatten                : PassThroughOutputDistributionEstimator,
            tfkeras.layers.GlobalAveragePooling1D : PoolingOutputDistributionEstimator,
            tfkeras.layers.GlobalAveragePooling2D : PoolingOutputDistributionEstimator,
            tfkeras.layers.GlobalAveragePooling3D : PoolingOutputDistributionEstimator,
            tfkeras.layers.GlobalMaxPooling1D     : PoolingOutputDistributionEstimator,
            tfkeras.layers.GlobalMaxPooling1D     : PoolingOutputDistributionEstimator,
            tfkeras.layers.GlobalMaxPooling1D     : PoolingOutputDistributionEstimator,
            tfkeras.layers.GRU                    : RecurrentOutputDistributionEstimator,
            tfkeras.layers.InputLayer             : PassThroughOutputDistributionEstimator,
            tfkeras.layers.LeakyReLU              : ActivationOutputDistributionEstimator,
            tfkeras.layers.LSTM                   : RecurrentOutputDistributionEstimator,
            tfkeras.layers.MaxPooling1D           : PoolingOutputDistributionEstimator,
            tfkeras.layers.MaxPooling2D           : PoolingOutputDistributionEstimator,
            tfkeras.layers.MaxPooling3D           : PoolingOutputDistributionEstimator,
            tfkeras.layers.Multiply               : MultiplyOutputDistributionEstimator,
            tfkeras.layers.Permute                : PassThroughOutputDistributionEstimator,
            tfkeras.layers.PReLU                  : ActivationOutputDistributionEstimator,
            tfkeras.layers.ReLU                   : ActivationOutputDistributionEstimator,
            tfkeras.layers.Reshape                : PassThroughOutputDistributionEstimator,
            tfkeras.layers.SimpleRNN              : RecurrentOutputDistributionEstimator,
            tfkeras.layers.SpatialDropout1D       : DropoutOutputDistributionEstimator,
            tfkeras.layers.SpatialDropout2D       : DropoutOutputDistributionEstimator,
            tfkeras.layers.SpatialDropout3D       : DropoutOutputDistributionEstimator,
            tfkeras.layers.Subtract               : SubtractOutputDistributionEstimator,
            tfkeras.layers.ThresholdedReLU        : ActivationOutputDistributionEstimator,
            tfkeras.layers.UpSampling1D           : PassThroughOutputDistributionEstimator,
            tfkeras.layers.UpSampling2D           : PassThroughOutputDistributionEstimator,
            tfkeras.layers.UpSampling3D           : PassThroughOutputDistributionEstimator,
            tfkeras.layers.ZeroPadding1D          : ZeroPaddingOutputDistributionEstimator,
            tfkeras.layers.ZeroPadding2D          : ZeroPaddingOutputDistributionEstimator,
            tfkeras.layers.ZeroPadding3D          : ZeroPaddingOutputDistributionEstimator,
            WeightedSum                           : WeightedSumOutputDistributionEstimator,
        }
        # Supported layers include the builtin layers and the user-provided ones.
        # {**dict1, **dict2} merges two dictionaries together
        self.supported_layers = {**self.builtin_layers, **custom_distribution_estimators}
        self.default_estimator = PassThroughOutputDistributionEstimator

        self.estimator_config = estimator_config

    def construct_estimation_method(self, layer: tfkeras.layers.Layer) \
                 -> LayerOutputDistributionEstimator:
        """
        This function constructs the estimation method.
        :param layer: TF/Keras Layer object which is used to determine the
        type of Estimator.
        :return estimator: If an estimator exists it is returned otherwise
        a default estimator is returned
        """
        layer_type = type(layer)
        try:
            estimator = self.supported_layers[layer_type](layer, self.estimator_config)
        except KeyError:
            logging.warning('No LayerOutputDistributionEstimator found for layer %s. Using the ' \
                'default %s instead.', layer_type.__name__, self.default_estimator.__name__)
            estimator = self.default_estimator(layer, self.estimator_config)
        return estimator
