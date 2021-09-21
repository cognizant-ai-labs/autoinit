# AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks

The AutoInit paper is available here: https://arxiv.org/abs/2109.08958

AutoInit analyzes your network's topology, layers, and activation functions and configures the network weights to ensure smooth signal propagation at initialization.  The weights are initialized according to the following recursive algorithm.

* Given input with mean `input_data_mean` and variance `input_data_variance`
* For all layers in the model
    * Get the mean and variance from all incoming layers
    * Use the incoming mean and variance, along with the layer type, to calculate the outgoing mean and variance 
    * If the layer has weights
        * Initialize the weights so that outgoing `mean = 0` and `variance = 1` (in expectation)
    * Return the outgoing mean and variance for consumption by downstream layers

## Usage
Install AutoInit with
```
pip install git+https://github.com/cognizant-ai-labs/autoinit.git
```
and ensure your TensorFlow version is at least 2.4.0.


Import the `AutoInit` class in your script
```python
from autoinit.initializer.initialize_weights import AutoInit
```
and use the class to wrap your model:
```python
training_model = AutoInit().initialize_model(training_model)
```
Here, `training_model` can be an already instantiated TensorFlow `Model` instance, or a dictionary config or JSON string defining the model.

## Options

The `AutoInit` class has the following optional arguments:
```python
class AutoInit(weight_init_config: Dict=None,
               input_data_mean: float=0.0,
               input_data_var: float=1.0)
```
The parameters `input_data_mean` and `input_data_var` can be specified if your data does not have zero mean and unit variance.  The `weight_init_config` dictionary is used to customize other aspects of AutoInit's behavior.  The following fields are available:

#### Distribution
```json
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

## Visualization

Use `autoinit/util/visualize_init.py` to visualize signal propagation across a default and AutoInit version of a given neural network.  The docstring to the `AutoInitVisualizer` class contains more information.

## Unsupported Layers

In order to correctly calculate the outgoing mean and variance, a `LayerOutputDistributionEstimator` must be created for every type of layer in a network.  Many layer types are already supported.  However, if your network has a layer which is not yet supported, you will see a warning message like the following.  

```
WARNING:root:No LayerOutputDistributionEstimator found for layer UpSampling2D. Using the default PassThroughOutputDistributionEstimator instead.
```

In this case, the mean and variance are simply returned unchanged.  If performance is satisfactory, no action is required.  Otherwise, you should create a `LayerOutputDistributionEstimator` for your specific layer type and validate its performance using `visualize_init.py`.

## Citation

If you use AutoInit in your research, please cite it using the following BibTeX entry
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
