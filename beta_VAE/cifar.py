import tensorflow as tf
from model import CVAE
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
from inception_score import Inception_score
from cifar_cls import compute_score

optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
mbs = tf.losses.MeanAbsoluteError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def reconstruction_loss(model, X):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1])
    return -tf.reduce_mean(logx_z)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test

def msq(pred, label):
    return tf.keras.losses.MeanSquaredError()(pred, label)


def ori_cross_loss(model, x, d):
    r_x = rotate(x, d)
    mean, logvar = model.encode(r_x)
    r_z = model.reparameterize(mean, logvar)
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, -s], [s, c]
    phi_z = rotate_vector(r_z, r_m)
    phi_x = model.decode(phi_z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1])

    return -tf.reduce_mean(logx_z)


def rota_cross_loss(model, x, d):
    r_x = rotate(x, d)
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    phi_z = rotate_vector(z, r_m)
    phi_x = model.decode(phi_z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=r_x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1])

    return -tf.reduce_mean(logx_z)

    #logx_z = cross_entropy(phi_x, r_x)



def compute_loss(model, x):
    beta = model.beta
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    '''
    reco_loss = reconstruction_loss(x_logit, x)
    kl_loss = kl_divergence(logvar, mean)
    beta_loss = reco_loss + kl_loss * beta
    '''
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logx_z + beta * (logpz - logqz_x))

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

def generate_and_save_images(model, epoch, test_input, file_path):
    mean, logvar = model.encode(test_input)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(12, 12))
    display_list = [test_input[0], predictions[0]]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    file_dir = './image/' + date + file_path
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig(file_dir +'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

def divide_dataset(train_data, train_labels, sample_size):
  labels = pd.DataFrame({'labels': train_labels.flatten()})
  dataset = []
  for i in range(0, 10):
    idx = labels[labels.labels == i].iloc[:sample_size].index
    train_images = train_data[idx]
    dataset.append(train_images)
  return np.array(dataset).reshape(10 * sample_size, 32, 32, 3)




def start_train(epochs, model, train_dataset, test_dataset, date, filePath):
    @tf.function
    def train_step(model, x, optimizer):
        for degree in range(0, 100, 10):
            d = np.radians(degree)
            with tf.GradientTape() as tape:
                r_x = rotate(x, d)
                ori_loss = compute_loss(model, x)
                rota_loss = reconstruction_loss(model, r_x)
                ori_cross_l = ori_cross_loss(model, x, d)
                rota_cross_l = rota_cross_loss(model, x, d)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    in_range_socres = []
    for i in range(0, 100, 10):
        theta = np.radians(i)
        fid, is_avg, is_std = compute_inception_score(model, theta)
        in_range_socres.append(is_avg)
    score = np.mean(in_range_socres)
    iteration = 0
    while (score <= 6.7):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
            iteration += 1
        end_time = time.time()
        epochs += 1
        in_range_socres = []
        for i in range(0, 100, 10):
            theta = np.radians(i)
            fid, is_avg, is_std = compute_inception_score(model, theta)
            in_range_socres.append(is_avg)
        score = np.mean(in_range_socres)
        #generate_and_save_images(model, epochs, test_sample, file_path)
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epochs + 1,
                                                    ckpt_save_path))
        compute_and_save_inception_score(model, file_path, iteration)
        print('Epoch: {}, time elapse for current epoch: {}'
              .format(epochs, end_time - start_time))
        print('The current score is {}'.format(score))

    #compute_and_save_inception_score(model, file_path)

def normalize(image):
  image = tf.cast(image, tf.float32)
  return image


def compute_inception_score(model, d):
    mean, logvar = model.encode(test_images)
    r_m = np.identity(model.latent_dim)
    z = model.reparameterize(mean, logvar)
    r_x = rotate(test_images, d)
    c, s = np.cos(d), np.sin(d)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    rota_z = matvec(tf.cast(r_m, dtype=tf.float32), z)
    phi_x = model.sample(rota_z)
    return compute_score(r_x, phi_x)


def compute_and_save_inception_score(model, filePath,iteration):
    start_time = time.time()
    best_fid, best_mean, base_std = compute_score(test_images, test_images)
    base_line_fid, base_line_mean, base_line_std = compute_inception_score(model, 0)
    result = []
    for i in range(0, 100, 10):
        theta = np.radians(i)
        in_range_fid, \
        in_range_inception_mean, \
        in_range_inception_std = compute_inception_score(model,  theta)
        result.append([in_range_fid, in_range_inception_mean, in_range_inception_std])
    fid, mean, std = np.mean(result, axis=0)
    df = pd.DataFrame({
            'iteration': iteration,
            "best_fid": best_fid,
            "best_mean": best_mean,
            "base_line_fid": base_line_fid,
            "base_line_mean": base_line_mean,
            "in_range_fid":fid,
            "in_range_mean": mean,
            "in_range_std": std,
        }, index=[1])
    file_dir = "./score/" + date + filePath
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.isfile(file_dir + '/inception_score.csv'):
        df.to_csv(file_dir +'/inception_score.csv')
    else:  # else it exists so append without writing the header
        df.to_csv(file_dir + '/inception_score.csv', mode='a', header=False)
    end_time = time.time()
    print("total compute inception time {}".format(end_time-start_time))



if __name__ == '__main__':
    (train_set, train_labels), (test_dataset, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_set = preprocess_images(train_set)
    test_images = preprocess_images(test_dataset)
    batch_size = 32
    latent_dim = 64
    epochs = 30
    batch_size = 32
    test_size = 10000
    date = '3_23/'
    for i in range(10, 0, -1):
        epochs = 0
        model = CVAE(latent_dim=latent_dim, beta=3, shape=[32, 32, 3])
        sample_size = i * 100
        train_size = sample_size * 10
        train_images = divide_dataset(train_set, train_labels, sample_size)
        #train_size = 10000
        #train_images = train_set
        batch_size = 32
        train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                         .shuffle(train_size).batch(batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(test_size).batch(batch_size))
        str_i = str(i)
        file_path = 'sample_test' + str_i
        start_train(epochs, model, train_dataset, test_dataset, date, file_path)

