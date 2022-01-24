# AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks

The AutoInit paper is available here: https://arxiv.org/abs/2109.08958


## Usage
Install AutoInit with
```
pip install git+https://github.com/cognizant-ai-labs/autoinit.git
```
and ensure your TensorFlow version is at least 2.4.0.


Import the `AutoInit` class in your script
```python
from autoinit import AutoInit
```
and use the class to wrap your model:
```python
training_model = AutoInit().initialize_model(training_model)
```
Here, `training_model` can be an already instantiated TensorFlow `Model` instance, or a dictionary config or JSON string defining the model.

## Algorithm
AutoInit analyzes your network's topology, layers, and activation functions and configures the network weights to ensure smooth signal propagation at initialization.  The weights are initialized according to the following recursive algorithm.

* Given input with mean `input_data_mean` and variance `input_data_variance`
* For all layers in the model
    * Get the mean and variance from all incoming layers
    * Use the incoming mean and variance, along with the layer type, to calculate the outgoing mean and variance 
    * If the layer has weights
        * Initialize the weights so that outgoing `mean = 0` and `variance = 1` (in expectation)
    * Return the outgoing mean and variance for consumption by downstream layers

## Options

The `AutoInit` class has the following optional arguments:
```python
class AutoInit(weight_init_config: Dict=None,
               custom_distribution_estimators: Dict=None,
               input_data_mean: float=0.0,
               input_data_var: float=1.0)
```
The parameters `input_data_mean` and `input_data_var` can be specified if your data does not have zero mean and unit variance.  The `custom_distribution_estimators` is useful for extending AutoInit to novel layer types, and is discussed in the "Unsupported Layers" section below.  Finally, the `weight_init_config` dictionary is used to customize other aspects of AutoInit's behavior.  The following fields are available:

#### Distribution
```python
# One of ['truncated_normal', 'untruncated_normal', 'uniform', 'orthogonal']
"distribution" : "truncated_normal",
```
AutoInit has mainly been tested with `"distribution" : "truncated_normal"`.  Other distribution choices should be validated experimentally or by using `visualize_init.py` (below).

#### Constrain Weights
```json
"constrain_weights" : false,
```
Setting `"constrain_weights" : true` constrains the model weights to always have zero empirical mean and maintain their scale *throughout training*, not just at initialization.  This setting was only used in early experiments.  Its effects are not well-understood, so it is disabled by default.

#### Monte Carlo Samples
```json
"monte_carlo_samples" : 1e7,
"recurrent_monte_carlo_samples" : 1e4,
```
Pooling and recurrent layers require Monte Carlo sampling to estimate the outgoing mean and variance.  The default number of samples can be adjusted if needed.

## Unsupported Layers

In order to correctly calculate the outgoing mean and variance, a `LayerOutputDistributionEstimator` must be created for every type of layer in a network.  Many layer types are already supported.  However, if your network has a layer which is not yet supported, you will see a warning message like the following.  

```
WARNING:root:No LayerOutputDistributionEstimator found for layer MyCustomLayer. Using the default PassThroughOutputDistributionEstimator instead.
```

In this case, the mean and variance are simply returned unchanged.  If performance is satisfactory, no action is required.  

As a toy example, consider a user-created layer called `MyCustomLayer` that simply doubles its input.  After applying this layer, the mean remains unchanged while the variance is multiplied by four.  This dynamic can be incorporated into AutoInit by creating a `LayerOutputDistributionEstimator` for `MyCustomLayer` and overriding the `estimate` method.

```python
from autoinit.estimators.estimate_layer_output_distribution import LayerOutputDistributionEstimator

class MyCustomOutputDistributionEstimator(LayerOutputDistributionEstimator):
    def estimate(self, means_in, vars_in):
        return means_in[0], 4.0 * vars_in[0]
```

To complete the example, two dictionaries are needed.  First, `custom_objects` maps layer names to layer classes and is needed for TensorFlow to serialize custom layers.  Second, `custom_distribution_estimators` maps layer classes to estimators so that AutoInit can correctly calculate the outgoing mean and variance.

```python
custom_objects = {"MyCustomLayer" : MyCustomLayer}
custom_distribution_estimators = {MyCustomLayer : MyCustomOutputDistributionEstimator}

training_model = AutoInit(custom_distribution_estimators=custom_distribution_estimators).initialize_model(training_model, custom_objects=custom_objects)

```

## Visualization

To plot signal propagation across a default and AutoInit version of a given neural network, use:
```python
from autoinit import AutoInitVisualizer
AutoInitVisualizer().visualize(training_model)
```
Visualization is useful for ensuring the initialization is behaving as expected, especially when using custom layers.  The docstring to the `AutoInitVisualizer` class in `autoinit/util/visualize_init.py` contains additional information.


## Citation

If you use AutoInit in your research, please cite it using the following BibTeX entry, and consider sharing your use case with us by emailing [bingham@cs.utexas.edu](mailto:bingham@cs.utexas.edu).
```
@misc{bingham2021autoinit,
      title={AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks}, 
      author={Garrett Bingham and Risto Miikkulainen},
      year={2021},
      eprint={2109.08958},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
