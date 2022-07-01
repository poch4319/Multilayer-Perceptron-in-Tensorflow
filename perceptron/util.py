import tensorflow as tf


def save(session: tf.compat.v1.Session,
         global_step: int) -> None:
    """ Save a model.

    session: Tensorflow session
    global_step: Global step count, used in the resulting filenames
    """
    saver = tf.compat.v1.train.Saver()

    saver.save(session,
               "perceptron",
               global_step=global_step)


def load(checkpoint_prefix) -> tf.compat.v1.Session:
    """ Load a model.

    checkpoint_prefix: The path to a checkpoint, excluding the suffixes (.index,
                       .meta or .data-*-*
    """
    session = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(f"{checkpoint_prefix}.meta")
    saver.restore(session, checkpoint_prefix)

    return session
