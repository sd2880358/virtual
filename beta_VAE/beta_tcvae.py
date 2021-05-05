import tensorflow as tf
from model import CVAE, Classifier
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
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    log_qz, logq_z_product = estimate_entropies(z, mean, logvar)
    tc = log_qz - logq_z_product
    kl_loss = kl_divergence(mean, logvar)

    return -tf.reduce_mean(logx_z + kl_loss + (beta-1) * tc)

def gaussian_log_density(samples, mean, logvar):
    pi = tf.constant(np.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-logvar)
    tmp = (samples - mean) * inv_sigma
    return -0.5 * (tmp * tmp + logvar + normalization)


def estimate_entropies(qz_samples, mean, logvar):
    log_q_z_prob = gaussian_log_density(
        tf.expand_dims(qz_samples,1),  tf.expand_dims(mean,0),
    tf.expand_dims(logvar, 0))
    log_q_z_product = tf.math.reduce_sum(
        tf.math.reduce_logsumexp(log_q_z_prob, axis=1, keepdims=False)
    )

    log_qz = tf.math.reduce_logsumexp(
        tf.math.reduce_sum(log_q_z_prob, axis=2, keepdims=False)
    )
    return log_qz, log_q_z_product


def generate_and_save_images(model, epoch, test_sample, file_path):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
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




def start_train(epochs, model, train_dataset, test_dataset, date, filePath):
    @tf.function
    def train_step(model, x, optimizer):
        with tf.GradientTape() as tape:
            ori_loss = compute_loss(model, x)
            total_loss = ori_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    generate_and_save_images(model, 0, test_sample, file_path)
    display.clear_output(wait=False)
    iteration = 0
    for epoch in range(epochs):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()
        loss = tf.keras.metrics.Mean()
        generate_and_save_images(model, epochs, test_sample, file_path)
        if (epoch + 1)%1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            for test_x in test_dataset:
                total_loss = compute_loss(model, test_x)
                loss(total_loss)
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epochs, elbo, end_time - start_time))


    #compute_and_save_inception_score(model, file_path)



def compute_mnist_score(model, classifier, z=0, d=0, r_m=0, initial=False):
    if (initial==True):
        mean, logvar = model.encode(test_images)
        r_m = np.identity(model.latent_dim)
        z = model.reparameterize(mean, logvar)
        d = np.radians(random.randint(0, 90))
    c, s = np.cos(d), np.sin(d)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    rota_z = matvec(tf.cast(r_m, dtype=tf.float32), z)
    phi_z = model.sample(rota_z)
    #fid = calculate_fid(test_images, phi_z)
    scores = classifier.mnist_score(phi_z)
    return scores


def calculate_fid(real, fake):
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def compute_and_save_mnist_score(model, classifier, epoch, filePath):
    in_range_socres = []
    mean, logvar = model.encode(test_images)
    r_m = np.identity(model.latent_dim)
    z = model.reparameterize(mean, logvar)
    fid_list = []
    for i in range(0, 100, 10):
        theta = np.radians(i)
        scores = compute_mnist_score(model, classifier, z, theta, r_m)
        in_range_socres.append(scores)
    in_range_mean, in_range_locvar = np.mean(in_range_socres), np.std(in_range_socres)
    df = pd.DataFrame({
        "in_range_mean":in_range_mean,
        "in_range_locvar": in_range_locvar,
    }, index=[epoch+1])
    file_dir = "./score/" + date + filePath
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.isfile(file_dir + '/filename.csv'):
        df.to_csv(file_dir +'/filename.csv')
    else:  # else it exists so append without writing the header
        df.to_csv(file_dir + '/filename.csv', mode='a', header=False)


if __name__ == '__main__':
    (train_set, train_labels), (test_dataset, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_set)
    test_images = preprocess_images(test_dataset)
    batch_size = 32
    latent_dim = 8
    num_examples_to_generate = 16
    test_size = test_images.shape[0]
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim=latent_dim, beta=6)
    sample_size = 1000
    epochs = 30

    train_size = sample_size * 10
    batch_size = 32

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                         .shuffle(len(train_images)).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(test_size).batch(batch_size))
    date = '5_5/'
    file_path = 'beta_tcvae/'
    start_train(epochs, model, train_dataset, test_dataset, date, file_path)


