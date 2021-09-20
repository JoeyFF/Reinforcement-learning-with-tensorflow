from keras import models
from keras import layers
from keras.datasets import mnist
import numpy as py


class Keras():
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        self.train_images = train_images.reshape((60000, 28 * 28)).astype('float32')/255
        self.train_labels = train_labels
        self.test_images = test_images.reshape((10000, 28 * 28)).astype('float32')/255
        self.test_labels = test_labels

    def train(self):
        network = models.Sequential()
        network.add(layers.Dense(512,activation='relu', input_shape=(28*28,)))
        network.add(layers.Dense(10, activation='softmax'))
        network.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        network.fit(self.train_images, self.train_labels, epochs=5, batch_size=128)


if __name__ == '__main__':
    myKeras = Keras()
    myKeras.train()