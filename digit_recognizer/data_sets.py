import os

import pandas as pd
import numpy as np

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

class DataSet:

    def read_data_sets(self, data_dir):
        # Import data
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))

        train_images = train_data.iloc[:, 1:].values
        train_images = train_images.astype(np.float)

        # convert from [0:255] => [0.0:1.0]
        train_images = np.multiply(train_images, 1.0 / 255.0)

        test_images = test_data.values.astype(np.float)
        self.test_images = np.multiply(test_images, 1.0 / 255.0)

        labels = train_data[[0]].values.ravel()

        self.validation_images = train_images[:VALIDATION_SIZE]
        self.validation_labels = labels[:VALIDATION_SIZE]

        self.train_images = train_images[VALIDATION_SIZE:]
        self.train_labels = labels[VALIDATION_SIZE:]

        self.num_examples = self.train_images.shape[0]
        self.index_in_epoch = 0
        self.epochs_completed = 0

    # convert class labels from scalars to one-hot vectors
    # 0 => [1 0 0 0 0 0 0 0 0 0]
    # 1 => [0 1 0 0 0 0 0 0 0 0]
    # ...
    # 9 => [0 0 0 0 0 0 0 0 0 1]
    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    # serve data by batches
    def next_batch(self, batch_size):

        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        # when all trainig data have been already used, it is reorder randomly
        if self.index_in_epoch > self.num_examples:
            # finished epoch
            self.epochs_completed += 1
            # shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.train_images[start:end], self.train_labels[start:end]

input_data = DataSet()