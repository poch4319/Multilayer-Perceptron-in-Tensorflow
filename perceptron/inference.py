#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from perceptron.data import batch_generator, Example, read_iris_data
from perceptron.util import load
from typing import List, Optional, Tuple


def inference_and_evaluation(session: tf.compat.v1.Session,
                             model_op: tf.Tensor,
                             data_set: List[Example],
                             input_placeholder: tf.Tensor,
                             label_map: List[str],
                             batch_size: int
                            ) -> Tuple[float, int]:
    """ Perform inference, return accuracy and error count.

              session: Tensorflow session with loaded weights
             model_op: As returned by perceptron.model.model()
             data_set: The data set to run inference on
    input_placeholder: Placeholder for the inputs
            label_map: List of labels, mapping labels to indices
           batch_size: The batch size
    """
    ###########################################################################
    # TODO
    ###########################################################################
    assert batch_size == len(data_set)
    
    label_placeholder = tf.compat.v1.placeholder("float",
                                                 shape=[None, len(label_map)],
                                                 name="Y")

    correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(label_placeholder, 1))
    correct_count = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    with session.as_default():
        for test_data in batch_generator(data_set,
                                         label_map,
                                         batch_size,
                                         randomize=False):
            batch_x, batch_y = test_data.feature_vectors, test_data.label_vectors
            
            count, acc = session.run([correct_count, accuracy], feed_dict={input_placeholder: batch_x, label_placeholder: batch_y})
            error_counts = int(len(data_set) - count)
            return (acc, error_counts)
        
if __name__ == "__main__":
    data, feature_map, label_map = read_iris_data("data/iris_flowers-test.csv",
                                                  normalization_type="none") # Needs to be identical to the setting the model was trained with
    # We're working in Tensorflow v1 mode, i.e. no eager execution
    tf.compat.v1.disable_eager_execution()

    session = load("./gold_checkpoint/perceptron-375")                         # Should give 100% accuracy on the test set

    # Get the input placeholder as well as the model output's tensor directly
    # from the graph
    input_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("X:0")
    model_op = tf.compat.v1.get_default_graph().get_tensor_by_name("layer-2:0")# That's the model's last layer (if it is a 2 layer MLP)

    accuracy, error_count = inference_and_evaluation(session,
                                                     model_op,
                                                     data,
                                                     input_placeholder,
                                                     label_map,
                                                     batch_size=len(data))
    print(f"Test {round(accuracy * 100, 1)}% ({error_count})")
