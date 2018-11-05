import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os
import glob
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import cv2
from keras.preprocessing.image import ImageDataGenerator
from random import randint
from sklearn.model_selection import train_test_split


# TODO: modify path
img_dir = "/content/gdrive/My Drive/STU.AI/camera/training_data_2nd/"
data = pd.read_csv(img_dir + 'label.csv')

X_data = []
Y_data = []

image_size = (28, 28)
data = shuffle(data)

for i in range(len(data)):
    filename = data.iloc[i, 0]
    label = data.iloc[i, 2]

    img = plt.imread(img_dir + filename)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_CUBIC)
    X_data.append(img)
    Y_data.append(label)

X_data = np.array(X_data)
Y_data = np.array(Y_data)
n_classes = int(max(Y_data)) + 1

rgb2gray = [0.299, 0.587, 0.114]
# Convert images to grayscale
X_gray = np.dot(X_data[...,:3], rgb2gray)

# Normalize images
X_gray = (X_gray/255-0.5)*2
X_gray = X_gray.reshape(*X_gray.shape, 1)

def dataAug(X_train, y_train):
    datagen = ImageDataGenerator(rotation_range=10,zoom_range=0.10)
    for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=X_train.shape[0], shuffle=False):
        X_train_aug = x_batch.astype('uint8')
        y_train_aug = y_batch
        break

    X_train = np.concatenate([X_train, X_train_aug])
    y_train = np.concatenate([y_train,y_train_aug])
    return X_train,y_train


def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 28x28x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(1, 1, 1, 6), mean=mu, stddev=sigma))
    conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.bias_add(conv1, b1)
    print("layer 1 shape:", conv1.get_shape())

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2 = tf.nn.conv2d(conv1, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.bias_add(conv2, b2)
    print("layer 2 shape:", conv2.get_shape())

    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    conv2 = flatten(conv2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    W3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    b3 = tf.Variable(tf.zeros(120))
    full1 = tf.add(tf.matmul(conv2, W3), b3)

    # TODO: Activation.
    full1 = tf.nn.relu(full1)

    # Dropout
    # full1 = tf.nn.dropout(full1, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    W4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    b4 = tf.Variable(tf.zeros(84))
    full2 = tf.add(tf.matmul(full1, W4), b4)

    # TODO: Activation.
    full2 = tf.nn.relu(full2)

    # Dropout
    # full2 = tf.nn.dropout(full2, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    W5 = tf.Variable(tf.truncated_normal(shape=(84, 14), mean=mu, stddev=sigma))
    b5 = tf.Variable(tf.zeros(14))
    logits = tf.add(tf.matmul(full2, W5), b5)

    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples, logits


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, (None, 28, 28, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    learning_rate = 0.001
    EPOCHS = 50
    BATCH_SIZE = 64

    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        X_train, X_valid, y_train, y_valid = train_test_split(X_gray, Y_data, test_size=0.3, random_state=50)

        X_train, y_train = dataAug(X_train, y_train)

        num_examples = len(X_train)

        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy, _ = evaluate(X_valid, y_valid)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))

        saver.save(sess, './LeNet')
