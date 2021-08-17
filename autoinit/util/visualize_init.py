
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

import matplotlib.pyplot as plt
import numpy as np

# TF uses a complicated LazyLoader that pylint cannot properly comprehend.
# See https://stackoverflow.com/questions/65271399/vs-code-pylance-pylint-cannot-resolve-import
import tensorflow.keras as tfkeras

from autoinit.initializer.initialize_weights import AutoInit


class AutoInitVisualizer:
    """
    This class produces plots which visualize the mean and variance of signals as they propagate
    through the layers of a network.  It is useful for getting an intuition for how the weight
    initialization scheme is performing for a given network.  The initialization scheme is
    performing well if the variance does not vanish nor explode, but remains close to one with
    network depth.  Checking whether the predicted mean and variance track the actual mean and
    variance reasonably well is also useful for debugging OutputDistributionEstimators.  Mean and
    variance for default network initialization are also plotted for comparison.  These networks
    may (or may not) suffer from vanishing or exploding variance, especially if the network does
    not have any normalization layers.  Note that although the plots produced are sequential
    (shallow layers are on the left and deeper layers on the right), the networks themselves may
    not be.  Do not rely on these plots for an understanding of your network's architecture.
    Instead, use a tool designed for this purpose such as SeeNN.
    """
    def __init__(self):
        self.default_model = None
        self.autoinit_model = None
        self.mean_var_estimates = None

        self.layer_names = []
        self.default_means = []
        self.default_vars = []
        self.actual_means = []
        self.actual_vars = []
        self.pred_means = []
        self.pred_vars = []


    def _get_accessible_outputs(self, model, warn=True):
        outputs = []
        for layer in model.layers:
            if isinstance(layer, tfkeras.Model):
                if warn:
                    logging.warning("Submodel %s outputs aren't accessible and won't be " \
                                "visualized.  Try using a flattened version of your Model " \
                                " that contains no submodels. " \
                                "(See https://github.com/tensorflow/tensorflow/issues/34977.)",
                                layer.name)
            else:
                outputs.append(layer.output)
        return outputs


    def _load_models(self, model, custom_objects=None):
        """
        This function creates a default initialized and universal initialized model.
        The argument model can either be an already instantiated TensorFlow model, a
        model config dictionary, or a model JSON string.
        """
        if custom_objects is None:
            custom_objects = {}

        # Start by loading the default model
        if isinstance(model, tfkeras.Model):
            self.default_model = model
        elif isinstance(model, dict):
            self.default_model = tfkeras.Model.from_config(model, custom_objects=custom_objects)
        elif isinstance(model, str):
            self.default_model = tfkeras.models.model_from_json(model,
                                                                custom_objects=custom_objects)
        else:
            raise TypeError('The model argument must be a TensorFlow Model, config dictionary, '\
                           f'or JSON string.  Got type {type(model)}.')

        # Create the universal model based on the default model
        self.autoinit_model, self.mean_var_estimates = \
            AutoInit().initialize_model(self.default_model.get_config(), True)

        # Expose intermediate outputs of both models
        self.default_model = tfkeras.Model(inputs=self.default_model.inputs,
                                           outputs=self._get_accessible_outputs(self.default_model))
        self.autoinit_model = tfkeras.Model(inputs=self.autoinit_model.inputs,
                            outputs=self._get_accessible_outputs(self.autoinit_model, warn=False))


    def _clear_data(self):
        """
        Clear any values that may be persisting from previous visualizations.
        """
        self.layer_names = []
        self.default_means = []
        self.default_vars = []
        self.actual_means = []
        self.actual_vars = []
        self.pred_means = []
        self.pred_vars = []


    def _pass_noise_through_models(self, num_samples):
        """
        Pass random noise through the networks to analyze how the
        signal mean and variance changes across network layers.
        """
        self._clear_data()
        input_noise = []
        for inpt in self.autoinit_model.inputs: # pylint: disable=not-an-iterable
            size = [num_samples] + list(inpt.shape[1:])
            input_noise.append(np.random.normal(loc=0.0, scale=1.0, size=size))

        default_outputs = self.default_model(input_noise, training=True)
        autoinit_outputs = self.autoinit_model(input_noise, training=True)

        non_submodel_layers = [layer for layer in self.autoinit_model.layers \
                            if not isinstance(layer, tfkeras.Model)]


        # Collect mean and variance estimates and calculations
        for default_output, autoinit_output, layer in zip(default_outputs,
                                                    autoinit_outputs,
                                                    non_submodel_layers):
            if hasattr(layer, 'activation'):
                self.layer_names.append(f'{layer.name} + {layer.activation.__name__}')
            else:
                self.layer_names.append(layer.name)

            self.default_means.append(np.mean(default_output))
            self.default_vars.append(np.var(default_output))
            self.actual_means.append(np.mean(autoinit_output))
            self.actual_vars.append(np.var(autoinit_output))
            pred_mean, pred_var = self.mean_var_estimates[layer.name]
            self.pred_means.append(pred_mean)
            self.pred_vars.append(pred_var)


    def _create_plots(self, plot_path):
        """
        Produces plots that show the predicted, actual, and default mean and variance
        of signals as they pass through the network layers.
        """
        xvals = list(range(len(self.layer_names)))
        fig_height = 10
        fig_width = len(xvals) / 4

        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(xvals, self.actual_means, label='Universal Init Mean (Actual)', color='C0')
        plt.plot(xvals, self.pred_means, label='Universal Init Mean (Predicted)',
            color='C0', linestyle=':')
        plt.plot(xvals, self.default_means, label='Default Init Mean', color='C1')
        plt.xticks(ticks=xvals, labels=self.layer_names, rotation=90, fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if plot_path:
            plt.savefig(f'{plot_path}_mean')
        else:
            plt.show()

        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(xvals, self.actual_vars, label='Universal Init Variance (Actual)', color='C0')
        plt.plot(xvals, self.pred_vars, label='Universal Init Variance (Predicted)',
            color='C0', linestyle=':')
        plt.plot(xvals, self.default_vars, label='Default Init Variance', color='C1')
        plt.xticks(ticks=xvals, labels=self.layer_names, rotation=90, fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if plot_path:
            plt.savefig(f'{plot_path}_variance')
        else:
            plt.show()


    def visualize(self, model, custom_objects=None, plot_path=None, num_samples=100):
        """
        This function is the entry point to this class.
        :param model: TensorFlow Model or a dictionary config or JSON string defining the model
        :param num_samples: Number of random inputs to pass through the network
        :param plot_path: Path where plots should be saved.  If not specified, plots are
            displayed with plt.show()
        :param custom_objects: A dictionary of custom objects such as Layers, Constraints,
            and so on, needed for the Model to be JSON-serializable
        """
        self._load_models(model, custom_objects)
        self._pass_noise_through_models(num_samples)
        self._create_plots(plot_path)
