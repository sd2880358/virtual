import tensorflow as tf
import numpy as np
from math import floor
class Generator(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim=100):
    super(Generator, self).__init__()
    self.latent_dim = latent_dim
    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(8*8*256, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(8, 8, 256)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
        pre = logits/127.5 + 0.5
        probs = tf.sigmoid(pre)
        return probs
    return logits

class Discriminator(tf.keras.Model):
    def __init__(self, shape):
        super(Discriminator, self).__init__()
        self.shape = shape
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.shape)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                # No activation
                tf.keras.layers.Dense(2, activation='softmax'),
            ]
        )

    @tf.function
    def result(self, x):
        mix_result = self.model(x)
        cat, fidelity = tf.split(mix_result, num_or_size_splits=2, axis=1)
        return cat, fidelity