import tensorflow as tf
import numpy as np
from math import floor
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, beta=4, shape=[28,28,1], model="cnn"):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.beta = beta
    self.shape = shape
    self.output_f = int(shape[0]/4)
    self.output_s = shape[2]
    if (model == "cnn"):
        self.encoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=shape),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=self.output_f * self.output_f *32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(self.output_f, self.output_f, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=self.output_s, kernel_size=3, strides=1, padding='same'),
            ]
        )
    elif (model == "mlp"):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=shape),
                tf.keras.layers.Dense(
                    32, activation='relu'),
                tf.keras.layers.Dense(
                64, activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(latent_dim * latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(
                    512, activation='relu'),
                tf.keras.layers.Dense(
                    2352, kernel_size=3,
                    activation='relu'),
                # No activation
                tf.keras.layers.Reshape(target_shape=(28*28*3)),
                tf.keras.layers.Dense(
                    1)
            ]
        )
        assert self.decoder.output_shape == (None, 28, 28, 1)

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

class Classifier(tf.keras.Model):
    def __init__(self, shape):
        super(Classifier, self).__init__()
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
                tf.keras.layers.Dense(10, activation='softmax'),
            ]
        )

    def call(self, X):
        return self.model(X)


    def mnist_score(self, X, n_split=10, eps=1E-16):
        scores = list()
        n_part = floor(X.shape[0] / n_split)
        for i in range(n_split):
            # retrieve images
            ix_start, ix_end = i * n_part, (i + 1) * n_part
            subset = X[ix_start:ix_end]
            # convert from uint8 to float32
            subset = tf.cast(subset, tf.float32)
            p_yx = self.model.predict(subset)
            # calculate p(y)
            p_y = np.expand_dims(p_yx.mean(axis=0), 0)
            # calculate KL divergence using log probabilities
            kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
            # sum over classes
            sum_kl_d = kl_d.sum(axis=1)
            # average over images
            avg_kl_d = np.mean(sum_kl_d)
            # undo the log
            is_score = np.exp(avg_kl_d)
            # store
            scores.append(is_score)
        # average across images
        return scores

