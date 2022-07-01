#!/usr/bin/env python3

import argparse
import tensorflow as tf

from perceptron.data import read_iris_data, split_data_set
from perceptron.model import model
# added
from perceptron.training import training

def main() -> None:
    argparser = argparse.ArgumentParser(description='The perceptron.')

    argparser.add_argument("--training-data", "-d",
                           type=str,
                           default="./data/iris_flowers-train.csv",
                           help="Path to the Iris flower training data set")
    argparser.add_argument("--test-data", "-T",
                           type=str,
                           default="./data/iris_flowers-test.csv",
                           help="Path to the Iris flower test data set")
    argparser.add_argument("--valid-ratio", "-t",
                           type=float,
                           default=0.1,
                           help="Size of the validation set (as a ratio of the training set)")
    argparser.add_argument("--epochs", "-e",
                           type=int,
                           default=100,
                           help="Number of epochs")
    argparser.add_argument("--batch-size", "-b",
                           type=int,
                           default=30,
                           help="Batch size")
    argparser.add_argument("--learning-rate", "-r",
                           type=float,
                           default=0.1,
                           help="Learning rate")
    argparser.add_argument("--normalize", "-n",
                           type=str,
                           default="none",
                           help="Normalize feature values to the range [0,1] (minmax) or standardize them (std)")
    argparser.add_argument("--layers", "-l",
                           type=int,
                           default=2,
                           help="Number of layers for the multilayer perceptron")
    argparser.add_argument("--dimensionality", "-D",
                           type=int,
                           default=8,
                           help="Size of the hidden layer (if num. layers > 1)")
    argparser.add_argument("--intermediate-activation", "-a",
                           type=str,
                           default="linear",
                           help="Activation function in intermediate layers")

    args = argparser.parse_args()

    # We're working in Tensorflow v1 mode, i.e. no eager execution
    tf.compat.v1.disable_eager_execution()

    # Read the training data
    data, feature_map, label_map = read_iris_data(args.training_data, args.normalize)
    training_set, valid_set = split_data_set(data, ratio=args.valid_ratio)
    print(f"Training set size: {len(training_set)}, validation set size: {len(valid_set)}")

    input_dim = len(feature_map)
    output_dim = len(label_map)
    if args.layers == 1:
        hidden_dim = output_dim                                                # ... for a single layer
    else:
        hidden_dim = args.dimensionality                                       # ... for multiple layers

    # Input placeholders
    input_placeholder = tf.compat.v1.placeholder("float",
                                                 [None, input_dim],
                                                 name="X")                     # The features, could be a batch
    label_placeholder = tf.compat.v1.placeholder("float",
                                                 shape=[None, output_dim],
                                                 name="Y")                     # The label vector(s), one-hot encoded


    # Build the model
    model_op, var_list = model(input_placeholder,
                              hidden_dim,
                              output_dim,
                              args.layers,
                              args.intermediate_activation)
    ###########################################################################
    # TODO
    training(args,
             model_op,
             training_set,
             valid_set,
             input_placeholder,
             label_placeholder,
             label_map)
    ###########################################################################

if __name__ == "__main__":
    main()
