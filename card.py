# check GPU
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dropout
import tensorflow as tf
tf.test.is_gpu_available(cuda_only=True)


# load library

# 2.load training images
path = "e:/images"
os.chdir(path)
files = []
for file in os.listdir():
    if file.endswith(".jpg"):
        files.append(file)
        print(file)

file = files[0]
image = cv2.imread(file,0)
dim = image.shape
# size = 400
# dim = (500, 1145)
height = dim[0]
width = dim[1]


def create_training_data(data_path):
    training_data = []

    # iterate over each image
    for image in os.listdir(data_path):
        # check file extention
        if image.endswith(".jpg"):
            try:
                data_path = pathlib.Path(data_path)
                full_name = str(pathlib.Path.joinpath(data_path, image))
                data = cv2.imread(str(full_name), 0)
                training_data.append([data])
            except Exception as err:
                print("an error has occured: ", err, str(full_name))

    # normalize data
    training_data = np.array(training_data)/255.
    # reshape
    training_data = np.array(training_data).reshape(-1, height, width)
    return training_data

# 3.build model


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # input layer
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(32, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(height*width, activation='sigmoid'),
            layers.Reshape((height, width))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 4.threshold


def model_threshold(autoencoder, x_train):
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    loss = tf.keras.losses.mse(decoded_imgs, x_train)
    threshold = np.mean(loss) + np.std(loss)
    return threshold

# 5. make inference


def sample_loss(autoencoder, file):
    # data = np.ndarray(shape=(1, height, width), dtype=np.float32)
    # individual sample
    # Load an image from a file
    data = cv2.imread(str(file), 0)
    # resize to make sure data consistency
    # resized_data = cv2.resize(data, (size, size))
    # nomalize img
    # normalized_data = resized_data.astype('float32') / 255.
    normalized_data = data.astype('float32') / 255.
    # test an image
    encoded = autoencoder.encoder(normalized_data.reshape(-1, size, size))
    decoded = autoencoder.decoder(encoded)
    loss = tf.keras.losses.mse(decoded, normalized_data)
    sample_loss = np.mean(loss) + np.std(loss)
    return sample_loss


# prepare files
data = create_training_data(path)
x_train = data[:-1]
x_test = data[-1:]

# build model
autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

history = autoencoder.fit(x_train, x_train,
                          epochs=40,
                          shuffle=True,
                          validation_data=(x_test, x_test))

# plot history
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# save and load a mode
autoencoder.save('./model/')
autoencoder = keras.models.load_model('./model/')

'''
Make inferences for pass and fail
'''
# make an inference
file = "e:/images/anomaly/template.jpg" # pass
file = "e:/images/card/anomaly/fail.jpg"    #fail
image = cv2.imread(file, 0)
plt.imshow(image, cmap='gray')

normalized_data = image.astype('float32') / 255.

# decode an image and calulate loss
encoded = autoencoder.encoder(normalized_data.reshape(-1, dim[0], dim[1]))
decoded = autoencoder.decoder(encoded)
loss = tf.keras.losses.mse(decoded, normalized_data)
sample_loss = np.mean(loss) + np.std(loss)
print(sample_loss)

# loss graph
os.chdir("e:/images")
sample_losses = []
for file in os.listdir():
    if file.endswith("jpg"):
        image = cv2.imread(file, 0) 
        normalized_data = image.astype('float32') / 255.

        # decode an image and calulate loss
        encoded = autoencoder.encoder(normalized_data.reshape(-1, dim[0], dim[1]))
        decoded = autoencoder.decoder(encoded)
        loss = tf.keras.losses.mse(decoded, normalized_data)
        sample_loss = np.mean(loss) + np.std(loss)
        sample_loss = round(sample_loss,4)
        print(sample_loss)
        sample_losses.append(sample_loss)

plt.hist(sample_losses)

import csv
with open('sample_losses.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], sample_losses))

# compare images

def diff(a, b):
    '''
    subtract differences between autoencoder and reconstructed image
    '''
    # autoencoder - reconstructed
    inv_01 = cv2.subtract(a, b)

    # reconstructed - autoencoder
    inv_02 = cv2.subtract(b, a)

    # combine differences
    combined = cv2.addWeighted(inv_01, 0.5, inv_02, 0.5, 0)
    return combined

# generate decoded image
decoded_image = decoded.numpy()
decoded_image = decoded_image.reshape(height, width)
decoded_image = (decoded_image*255).astype(np.uint8)
plt.imshow(decoded_image, cmap='gray')

# a = decoded.numpy()
difference = diff(image,decoded_image)
plt.imshow(difference, cmap='magma')


