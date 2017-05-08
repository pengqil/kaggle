"""Trains and Evaluates the MNIST network using a feed dictionary."""
import argparse
import os.path
import sys
import time

import tensorflow as tf
import numpy as np
import mnist
from data_sets import input_data

# Basic model parameters as external flags.
FLAGS = None


def placeholder_inputs():
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(None ,mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32)
    return images_placeholder, labels_placeholder


def fill_feed_dict(train_images, train_labels, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    feed_dict = {
        images_pl: train_images,
        labels_pl: train_labels,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            images, labels):
    """Runs one evaluation against the full epoch of data.
    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    feed_dict = fill_feed_dict(images, labels,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / len(labels)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (len(labels), true_count, precision))


def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.
    input_data.read_data_sets(FLAGS.input_data_dir)

    # Tell TensorFlow that the model will be built into the default Graph.
    # with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs()

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(FLAGS.max_steps):
        start_time = time.time()

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        train_images, train_labels = input_data.next_batch(FLAGS.batch_size)
        feed_dict = fill_feed_dict(train_images, train_labels,
                                   images_placeholder,
                                   labels_placeholder)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict)

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)
            # Evaluate against the training set.
            print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    train_images, train_labels)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    input_data.validation_images, input_data.validation_labels)
    predicts = mnist.prediction(logits)
    predicted_labels = sess.run(predicts, {images_placeholder: input_data.test_images})
    np.savetxt('submission/submission_softmax_3.csv',
               np.c_[range(1, len(input_data.test_images) + 1), predicted_labels],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=20000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='log',
        help='Directory to put the log data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)