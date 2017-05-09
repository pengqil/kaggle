import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from data_sets import input_data

# Generate input data
input_data.read_data_sets("data")
train_data = input_data.train_images
train_labels = input_data.train_labels_onehot
eval_data = input_data.validation_images
eval_labels = input_data.validation_labels_onehot

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Reshape((28, 28, 1), input_shape=(784,)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(train_data, train_labels, batch_size=100, epochs=10)
score = model.evaluate(eval_data, eval_labels)

predicted_labels = model.predict(input_data.test_images).argmax(axis=-1)
np.savetxt('submission/submission_softmax_5.csv',
           np.c_[range(1, len(input_data.test_images) + 1), predicted_labels],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')