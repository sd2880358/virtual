import tensorflow as tf
from model import CVAE, Classifier, SIM_CLR
from dataset import preprocess_images, divide_dataset
from tensorflow_addons.image import rotate
import time
from tensorflow.linalg import matvec
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import math

optimizer = tf.keras.optimizers.Adam(1e-4)
mbs = tf.losses.MeanAbsoluteError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def reconstruction_loss(model, X, y):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    h = model.projection(Z)
    encode_loss = top_loss(model, h, y)
    return -tf.reduce_mean(logx_z) + encode_loss, h


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test


def ori_cross_loss(model, x, d, r_x):
    mean, logvar = model.encode(r_x)
    r_z = model.reparameterize(mean, logvar)
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, -s], [s, c]
    phi_z = rotate_vector(r_z, r_m)
    phi_x = model.decode(phi_z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)


def rota_cross_loss(model, x, d, r_x):
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    phi_z = rotate_vector(z, r_m)
    phi_x = model.decode(phi_z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=r_x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)




def compute_loss(model, x, y):
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
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    h = model.projection(z)
    encode_loss = top_loss(model, h, y)
    return -tf.reduce_mean(logx_z + beta * (logpz - logqz_x)) + encode_loss, h


def top_loss(model, h, y):
    classes = model.num_cls
    labels = tf.one_hot(y, classes)
    loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=h
    ))

    return loss_t




def start_train(epochs, model, train_set, test_set, date, filePath):
    @tf.function
    def train_step(model, x, y, degree_set, optimizer):
        s = degree_set[0]
        e = degree_set[1]
        for i in range(s, e + 10, 10):
            d = np.radians(i)
            with tf.GradientTape() as tape:
                r_x = rotate(x, d)
                ori_loss, _ = compute_loss(model, x, y)
                rota_loss, _ = reconstruction_loss(model, r_x, y)
                ori_cross_l = ori_cross_loss(model, x, d, r_x)
                rota_cross_l = rota_cross_loss(model, x, d, r_x)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(sim_clr=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)
    for epoch in range(epochs):

        start_time = time.time()

        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            train_step(model, x, y, [0, 360], optimizer)

        end_time = time.time()
        loss = tf.keras.metrics.Mean()

        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch + 1)%10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            for i in range(0, 370, 10):
                d = np.radians(i)
                r_x = rotate(test_set[0], d)
                ori_loss, _ = compute_loss(model, test_set[0], test_set[1])
                rota_loss, r_h = reconstruction_loss(model, test_set[0], test_set[1])
                ori_cross_l = ori_cross_loss(model, test_set[0], d, r_x)
                rota_cross_l = rota_cross_loss(model, test_set[0], d, r_x)
                correct_r_h = np.sum(r_h.numpy().argmax(-1) == test_labels)
                acc = (correct_r_h/float(len(test_labels)))
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
                loss(total_loss, acc)
            elbo = loss.result()
            print('Epoch: {}, elbo: {}, accuracy: {}, time elapse for current epoch: {}'
                  .format(epoch+1, -elbo[0], elbo[1], end_time - start_time))

    #compute_and_save_inception_score(model, file_path)


def divide_array(l):
    m = int(math.floor(len(l))/2)
    head = l[:m]
    tail = l[m:m*2]
    return head, tail

def pre_cast(dataset, digits):
    tmp =  dataset[np.where(np.isin(mnist_labels, [digits[0]]))]
    head, tail  = divide_array(tmp)
    labels = np.array([0]*len(head))
    for i in range(1, len(digits)-1):
        array = mnist_images[np.where(np.isin(mnist_labels, [digits[i]]))]
        h, t = divide_array(array)
        l = np.array((i) * len(h))
        head = np.concatenate([head, h])
        tail = np.concatenate([tail, t])
        labels = np.concatenate([labels, l])
    return head, tail,labels



if __name__ == '__main__':
    (mnist_images, mnist_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images)
    test_images = preprocess_images(test_images)
    train_images = mnist_images[np.where(np.isin(mnist_labels, [0, 1]))]
    test_images = test_images[np.where(np.isin(test_labels, [0, 1]))]
    train_labels = mnist_labels[np.isin(mnist_labels, [0,1])]
    train_labels = test_labels[np.isin(test_labels, [0, 1])]
    num_examples_to_generate = 16
    model = CVAE(latent_dim=8, beta=6, shape=[28, 28, 1])
    epochs = 50
    batch_size = 32
    sim_clr = SIM_CLR()
    train_images = (tf.data.Dataset.from_tensor_slices(mnist_images)
            .shuffle(len(mnist_images), seed=1).batch(batch_size))

    train_labels = (tf.data.Dataset.from_tensor_slices(mnist_labels)
                    .shuffle(len(mnist_labels), seed=1).batch(batch_size))


    date = '6_16/'
    file_path = 'clr_test2/'
    start_train(epochs, sim_clr, [train_images, train_labels], [test_images, test_labels], date, file_path)