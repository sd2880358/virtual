import tensorflow as tf
from model_v3 import CVAE
from dataset import preprocess_images, divide_dataset
from tensorflow_addons.image import rotate
import random
import time
from tensorflow.linalg import matvec
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import pandas as pd
from scipy.linalg import sqrtm

optimizer = tf.keras.optimizers.Adam(1e-4)
mbs = tf.losses.MeanAbsoluteError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def kl_divergence(mean, logvar):
    summand = tf.math.square(mean) + tf.math.exp(logvar) - logvar  - 1
    return (0.5 * tf.reduce_sum(summand, [1]))

def compute_loss(model, x):
    beta = model.beta
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    identity = model.decode(z)
    x_logit = model.reshape(identity)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))
    log_qz, logq_z_product = estimate_entropies(z, mean, logvar)
    tc = tf.reduce_mean(log_qz - logq_z_product)
    kl_loss = tf.reduce_mean(kl_divergence(mean, logvar))
    return tf.reduce_mean(logx_z + kl_loss + (beta-1) * tc)

def rec_loss(x, y):
    r_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    log_r_x_z = tf.reduce_sum(r_cross_ent, axis=[1, 2, 3])
    return tf.reduce_mean(log_r_x_z)

def gaussian_log_density(samples, mean, logvar):
    pi = tf.constant(np.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-logvar)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)

def further_dis(model, x, r_x, radians):
    x_angle = 0
    rx_angle = radians

    x_mean, x_logvar = model.encode(x)
    y = model.reparameterize(x_mean, x_logvar)
    rx_logit = model.decode(rx_angle, y)

    rx_mean, rx_logvar = model.encode(r_x)
    r_y = model.reparameterize(rx_mean, rx_logvar)
    x_logit = model.decode(x_angle, r_y)

    logx_z =  rec_loss(x_logit, x)
    logrx_z = rec_loss(rx_logit, r_x)

    return tf.reduce_mean(logx_z + logrx_z)

def reconstruction_loss(model, X):
    mean, logvar = model.encode(X)
    Y = model.reparameterize(mean, logvar)
    X_pred = model.decode(0, Y)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return -tf.reduce_mean(logx_z)


def estimate_entropies(qz_samples, mean, logvar):
    log_q_z_prob = gaussian_log_density(
        tf.expand_dims(qz_samples,1),  tf.expand_dims(mean,0),
    tf.expand_dims(logvar, 0))

    log_q_z_product = tf.math.reduce_sum(
        tf.math.reduce_logsumexp(log_q_z_prob, axis=1, keepdims=False),
        axis=1, keepdims=False
    )

    log_qz = tf.math.reduce_logsumexp(
        tf.math.reduce_sum(log_q_z_prob, axis=2, keepdims=False)
    )
    return log_qz, log_q_z_product



def ori_cross_loss(model, x, r_x):
    mean, logvar = model.encode(r_x)
    Y = model.reparameterize(mean, logvar)
    phi_x = model.decode(0, Y)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)


def rota_cross_loss(model, x, d, r_x):
    x_mean, x_logvar = model.encode(x)
    y = model.reparameterize(x_mean, x_logvar)

    phi_identity = model.decode(d, y)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_identity, labels=r_x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)


def compute_loss(model, x):
    beta = model.beta
    mean, logvar = model.encode(x)
    y = model.reparameterize(mean, logvar)
    x_logit = model.decode(0, y)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))
    log_qz, logq_z_product = estimate_entropies(y, mean, logvar)
    tc = tf.reduce_mean(log_qz - logq_z_product)
    kl_loss = tf.reduce_mean(kl_divergence(mean, logvar))

    return tf.reduce_mean(logx_z + kl_loss + (beta-1) * tc)



def generate_and_save_images(model, epoch, test_sample, file_path):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(0, z)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    file_dir = './image/' + date + file_path
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig(file_dir +'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def start_train(epochs, model, full_range_set, partial_range_set, date, filePath):
    @tf.function
    def train_step(x, degree_set):
        for i in range(10, degree_set + 10, 10):
            d = np.radians(i)
            with tf.GradientTape() as tape:
                r_x = rotate(x, d)
                ori_loss = compute_loss(model, x)
                rota_loss = reconstruction_loss(model, r_x)
                ori_cross_l = ori_cross_loss(model, x, r_x)
                rota_cross_l = rota_cross_loss(model, x, d, r_x)
                lz_loss = further_dis(model, x, r_x, d)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l + lz_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for test_batch in partial_range_set.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    generate_and_save_images(model, 0, test_sample, file_path)
    display.clear_output(wait=False)

    for epoch in range(epochs):
        start_time = time.time()

        for train_x in full_range_set:
            train_step(train_x, degree_set=360)

        for train_p in partial_range_set:
            train_step(train_p, degree_set=180)

        end_time = time.time()
        model_loss = tf.keras.metrics.Mean()
        decoder_loss = tf.keras.metrics.Mean()
        if (epoch + 1)%5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            for i in range(0, 390, 10):
                d = np.radians(i)
                r_x = rotate(test_sample, d)
                ori_loss = compute_loss(model, test_sample)
                rota_loss = reconstruction_loss(model, r_x)
                ori_cross_l = ori_cross_loss(model, test_sample, r_x)
                rota_cross_l = rota_cross_loss(model, test_sample, d, r_x)
                total_loss = rota_loss + ori_cross_l + rota_cross_l
                decoder_loss(total_loss)
                model_loss(ori_loss)

            elbo = -model_loss.result()
            decoder_loss = -decoder_loss.result()
            generate_and_save_images(model, epoch, test_sample, file_path)
            print('Epoch: {}, Decoder{}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch+1, elbo, decoder_loss, end_time - start_time))


    #compute_and_save_inception_score(model, file_path)





def calculate_fid(real, fake):
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid




if __name__ == '__main__':
    (mnist_images, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images)

    full_range = mnist_images[np.where(np.isin(mnist_labels, [4,5,6]))]
    partial_range = mnist_images[np.where(mnist_labels == 3)]
    num_examples_to_generate = 16
    model = CVAE(latent_dim=7, beta=6, shape=[28, 28, 1], model='cnn')
    epochs = 50

    batch_size = 32

    full_range_digit = (tf.data.Dataset.from_tensor_slices(full_range)
                         .batch(batch_size))
    partial_range_digit = (tf.data.Dataset.from_tensor_slices(partial_range)
                         .batch(batch_size))


    date = '5_31/'
    file_path = 'mnist_test12/'
    start_train(epochs, model, full_range_digit, partial_range_digit, date, file_path)


