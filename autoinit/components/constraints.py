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


class CenteredUnitNorm(tfkeras.constraints.Constraint):
    """
    Constrains the weights to have empirical zero mean and unit variance.  Use
    axis=0 for a Dense layer,
    axis=[0, 1] for a Conv1D layer,
    axis=[0, 1, 2] for a Conv2D layer, and
    axis=[0, 1, 2, 3] for a Conv3D layer.
    """

    def __init__(self, axis=0, gain=1.0, mode='fan_in'):
        """
        The constructor initializes the parameters
        """
        self.axis = axis
        self.gain = gain
        self.mode = mode

    def _compute_fans(self, shape):
        """
        Computes the number of input and output units for a weight shape.
        :param shape: Integer shape tuple or TF tensor shape.
        :return (fan_in, fan_out): A tuple of integer scalars.
        See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py.
        """
        if len(shape) < 1:  # Just to avoid errors for constants.
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assuming convolution kernels (2D, 3D, or more).
            # kernel shape: (..., input_depth, depth)
            input_depth_idx = -2
            depth_idx = -1
            receptive_field_size = 1
            # Iterate over receptive field dimensions (exclude input_depth and depth)
            for dim in shape[:-2]:
                receptive_field_size *= dim
            fan_in = shape[input_depth_idx] * receptive_field_size
            fan_out = shape[depth_idx] * receptive_field_size
        return int(fan_in), int(fan_out)

    # pylint: disable=invalid-name
    def __call__(self, w):
        """
        Override the call function.
        """
        fan_in, fan_out = self._compute_fans(w.shape)
        if self.mode == 'fan_in':
            neurons = fan_in
        elif self.mode == 'fan_out':
            neurons = fan_out
        else:
            neurons = (fan_in + fan_out) / 2.0

        return self.gain * (w - tfkeras.backend.mean(w)) / (
            tfkeras.backend.sqrt(neurons * tfkeras.backend.var(w, axis=self.axis, keepdims=True)))

    def get_config(self):
        """
        Serialize the constraint.
        """
        return {
            'axis': self.axis,
            'gain': self.gain,
            'mode': self.mode
        }
