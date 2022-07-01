import tensorflow as tf

from enum import Enum
from typing import List, Tuple, Optional


class ActivationType(Enum):
    TANH    = "tanh"
    RELU    = "relu"
    SIGMOID = "sigmoid"
    LINEAR  = "linear"


def model(input_placeholder: tf.Tensor,
          hidden_dim: int,
          output_dim: int,
          num_layers: int,
          activation: ActivationType
         ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """ The model, a multi-layer perceptron.

    input_placeholder: Placeholder for inputs
           hidden_dim: Dimensionality of hidden layer(s)
           output_dim: Dimensionality of the output space
           num_layers: # of layers
           activation: Activation function for intermediate layers
    """
    input_dim = int(input_placeholder.shape[1])
    if num_layers == 1:
        hidden_dim = output_dim
    var_list = []
    layer = None
    previous_layer = input_placeholder
    initializer = tf.compat.v1.truncated_normal_initializer(mean=0.0,
                                                            stddev=1.0,
                                                            dtype=tf.dtypes.float32)
    for n in range(num_layers):
        W = tf.Variable(initializer(shape=[input_dim, hidden_dim]),
                        name=f"W{n + 1}")                                      # The weight matrix for this layer
        b = tf.Variable(initializer(shape=[hidden_dim]),                       # Output bias for this layer
                        name=f"b{n + 1}")
        var_list.extend([W, b])                                                # Collect all variables
        layer_name = f"layer-{n + 1}"
        previous_layer = tf.add(tf.matmul(previous_layer, W),
                                b,
                                name=layer_name)
        if n < num_layers - 1:
            if activation == ActivationType.TANH:
                previous_layer = tf.tanh(previous_layer, name=layer_name)
            elif activation == ActivationType.RELU:
                previous_layer = tf.nn.relu(previous_layer, name=layer_name)
            elif activation == ActivationType.SIGMOID:
                previous_layer = tf.nn.sigmoid(previous_layer, name=layer_name)
            else:
                pass                                                           # Linear activation

        input_dim = hidden_dim
        if n + 1 == num_layers - 1:
            hidden_dim = output_dim

    return previous_layer, var_list
