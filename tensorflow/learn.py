# learn.py - This script learns a new model based on the imdb reviews test set.
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np

print("Tensorflow Version", tf.__version__)

# Load the imdb reviews dataset
imdb = keras.datasets.imdb

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

# Load both labeled test and train datasets. We limit the amount of used words to the first 10000 words.
# The limit also needs to be used in any code that uses the model for predictions.

# We receive a numpy array containing integers (will be later mapped to actual words)
# data = array of words
# label = classification bad/good (0/1)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

#  Present some parts of the datasets, not actually required for learning the model
print("Training entries: {}, test entries: {}".format(len(train_data), len(test_data)))
print("Training data example: {}, with label: {}".format(train_data[0], train_labels[0]))
print("Training data labels example", np.unique(train_labels))

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved for technical use
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown, any word that is not inside the dictionary
word_index["<UNUSED>"] = 3

# Create a reverse index to show the contents of the learning data
# This is not actually required for learning the model
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# Creates something like:
# "<START> this film was just brilliant casting <UNK> <UNK> story
#  direction <UNK> really <UNK> the part they played and you could just
#  imagine being there robert <UNK> is an amazing actor ..."


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print("Review example", decode_review(train_data[0]))

# Take the train and test reviews and pad them to a fixed size
# Here, we add 0 = word_index[<PAD>] to the array at the end
# when the length is shorter than 256
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print("Size of padded train entry", len(train_data[0]))
print("Padded train entry example", train_data[0])

# Create the model

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print("Model", model.summary())

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# Split into training and validation data
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Perform the training
# epochs are set to 20 to avoid overfitting in this particular example
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the trained model
results = model.evaluate(test_data, test_labels)
# Returns loss and metric, e.g. it has an accuracy of 88 %
print("Model Evaluation", results)

# Save the whole model, not only the weights
model.save('reviews.h5')
print('Stored as reviews.h5')
