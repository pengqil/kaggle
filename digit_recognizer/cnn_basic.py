import os

import tensorflow as tf
import pandas as pd
import numpy as np

# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 20000

DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# Import data
data_path = "data"
train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
test_data = pd.read_csv(os.path.join(data_path, "test.csv"))

train_images = train_data.iloc[:, 1:].values
train_images = train_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
train_images = np.multiply(train_images, 1.0 / 255.0)

test_images = test_data.values.astype(np.float)
test_images = np.multiply(test_images, 1.0 / 255.0)

image_size = train_images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

labels_flat = train_data[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]


# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

validation_images = train_images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = train_images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


# Create the model
x = tf.placeholder(tf.float32, [None, image_size])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# First Layer
W_conv1 = weight_variable([3, 3, 1, 32])  # The convolution will compute 32 features for each 5x5 patch.
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, image_width, image_height, 1])  # the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Second Layer
W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_conv3, [-1, 7*7*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, labels_count])

# Evaluation
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y_conv, 1)


# Train, Valid and Predict
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]


# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


# start TensorFlow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.run(tf.global_variables_initializer())

display_step = 1

for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        if VALIDATION_SIZE:
            validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                           y_: validation_labels,
                                                           keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
                train_accuracy, validation_accuracy, i))
        else:
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
        # increase display_step
        if i % (display_step * 10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})


# predict test set
predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})
np.savetxt('submission/submission_softmax_2.csv',
           np.c_[range(1, len(test_images)+1), predicted_lables],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')
