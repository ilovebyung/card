# 1.load library
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dropout
import tensorflow as tf
tf.test.is_gpu_available(cuda_only=True)

# 2.load image dimension
file = "/home/byungsoo/Documents/card/images/roi_00.jpg"
image = cv2.imread(file,0)
dim = image.shape
height = dim[0]
width = dim[1]

# 3.load model
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
    
autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder = keras.models.load_model('/home/byungsoo/Documents/card/model/')

# 4.utility functions
def sample_loss(file):
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
    encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width))
    decoded = autoencoder.decoder(encoded)
    loss = tf.keras.losses.mse(decoded, normalized_data)
    sample_loss = np.mean(loss) + np.std(loss)
    return sample_loss

def calulate_loss(image):
    normalized_data = image.astype('float32') / 255.
    # decode an image and calulate loss
    encoded = autoencoder.encoder(normalized_data.reshape(-1, dim[0], dim[1]))
    decoded = autoencoder.decoder(encoded)
    loss = tf.keras.losses.mse(decoded, normalized_data)
    sample_loss = np.mean(loss) + np.std(loss)
    print(sample_loss)
    return round(sample_loss,4)

def generate_decoded(image):
    # generate decoded image
    normalized_data = image.astype('float32') / 255.
    # decode an image and calulate loss
    encoded = autoencoder.encoder(normalized_data.reshape(-1, dim[0], dim[1]))
    decoded = autoencoder.decoder(encoded)
    decoded_image = decoded.numpy()
    decoded_image = decoded_image.reshape(height, width)
    decoded_image = (decoded_image*255).astype(np.uint8)
    # plt.imshow(decoded_image, cmap='gray')
    return decoded_image

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


if __name__ == "__main__":
    '''
    Make inferences for pass and fail
    '''
    # make an inference
    file = "/home/byungsoo/Documents/card/images/anomaly/template.jpg" # pass
    file = "/home/byungsoo/Documents/card/images/anomaly/fail.jpg"    #fail
  
    image = cv2.imread(file, 0)
    plt.imshow(image, cmap='gray')

    sample_loss = sample_loss(file)
    print(sample_loss)

    # compare images
    decoded_image = generate_decoded(image)

    # a = decoded.numpy()
    difference = diff(image,decoded_image)
    plt.imshow(difference, cmap='magma')






