import tensorflow as tf
from perceptron.data import batch_generator, Example
from perceptron.inference import inference_and_evaluation
from perceptron.data import read_iris_data



from typing import List

def training(args,
             model_op: tf.Tensor,
             training_set: List[Example],
             valid_set: List[Example],
             X:  tf.Tensor,
             Y:  tf.Tensor,
             label_map: List[str]) -> None:
    """ The training loop.
    """
    ###########################################################################
    # TODO
    ###########################################################################
    
    # parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    
    # loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_op, labels=Y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) 
    train_op = optimizer.minimize(loss_op)
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()
    
    def training_step_fn() -> None:
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        
    with tf.compat.v1.Session() as sess:
        
        # Run the initializer
        sess.run(init)
        
        for epoch in range(epochs):
            for training_batch in batch_generator(training_set,
                                                label_map,
                                                batch_size,
                                                randomize=True):
                batch_x, batch_y = training_batch.feature_vectors, training_batch.label_vectors
                # Run optimization
                training_step_fn()
                
            # Validation for each epochs using test and valid set
            train_acc, train_error_counts = inference_and_evaluation(sess,
                                                                    model_op,
                                                                    training_set,
                                                                    X,
                                                                    label_map,
                                                                    len(training_set))
            valid_acc, valid_error_counts = inference_and_evaluation(sess,
                                                                    model_op,
                                                                    valid_set,
                                                                    X,
                                                                    label_map,
                                                                    len(valid_set))
            print(f"Epoch {epoch + 1} | Training set: {round(train_acc * 100, 1)}% ({train_error_counts}) Validation set: {round(valid_acc * 100, 1)}% ({valid_error_counts})")
            
            # Early stopping strategy
            if round(valid_acc * 100) > 98 and round(train_acc * 100) > 98:
                if epoch + 1 < epochs:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
            
        # Test Set Evaluation
        test_set, _, _ = read_iris_data(args.test_data, args.normalize)
        test_acc, test_error_counts = inference_and_evaluation(sess,
                                                            model_op,
                                                            test_set,
                                                            X,
                                                            label_map,
                                                            len(test_set))
        print(f"Test set accuracy {round(test_acc * 100, 1)}% ({test_error_counts})")

        print("Training Finished!")

