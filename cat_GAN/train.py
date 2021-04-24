import tensorflow as tf
from model import Generator, Discriminator
from dataset import load_celeba
from tensorflow_addons.image import rotate
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import pandas as pd
from celebA import CelebA
from tensorflow_probability import distributions as tfd


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(pred_x):
    image_loss = cross_entropy(tf.ones_like(pred_x), pred_x)
    return image_loss


def discriminator_loss(pred_x, act_x):
    image_loss_r = cross_entropy(tf.ones_like(act_x), act_x)
    image_loss_p = cross_entropy(tf.zeros_like(pred_x), pred_x)
    return image_loss_r + image_loss_p


def generate_and_save_images(model, epoch, test_input, test_label, file_path):
    noise, n_lables = sample(1, generator.latent_dim, test_label)
    prediction = model.decode(noise)
    fig = plt.figure(figsize=(12, 12))
    display_list = [test_input[0]*0.5+0.5, prediction[0]*0.5+0.5]
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


def sample(size, latent_dim, true_label):
    z = tfd.Uniform(low=-1.0, high=1.0).sample([size, latent_dim-1])
    p_labels = np.array([1 if true_label[j]==0 else 0 for j in range((true_label.shape[0]))])
    p_labels = p_labels.reshape(true_label.shape[0], 1)
    test = tf.cast(p_labels, tf.float32)
    noise = tf.concat([z, test], axis=-1)
    return noise, p_labels


def start_train(epochs, generator, discriminator,
                gen_optimizer, disc_optimizer,
                train_dataset, train_labels,
                test_dataset, test_labels,
                date, filePath):
    
    @tf.functions
    def train_step(generator, discriminator, train_data,
                   train_label, gen_optimizer, disc_optimizer):
        noise, n_lables = sample(train_data.shape[0], generator.latent_dim, train_label)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_image = generator.decode(noise)
            fake_output  = discriminator.result(fake_image)
            true_output = discriminator.result(train_data)
            gen_loss = generator_loss(fake_output, n_lables)
            disc_loss = discriminator_loss(fake_output, true_output, train_label)
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(generator=generator,
                               discriminator = discriminator,
                               gen_optimizer=gen_optimizer,
                               disc_optimizer = disc_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for test_batch, label_batch in tf.data.Dataset.zip((test_dataset.take(1),
                                                       test_labels.take(1))):
        test_sample = test_batch[:1, :, :, :]
        test_label = label_batch[:1]
    generate_and_save_images(generator, 0, test_sample, test_label, file_path)
    display.clear_output(wait=False)
    for epoch in range(epochs):
        start_time = time.time()
        for train_x, train_y in tf.data.Dataset.zip((train_dataset, train_labels)):
            train_step(generator, discriminator, train_x, train_y, gen_optimizer, disc_optimizer)
        generate_and_save_images(generator, epoch, test_sample, test_label, file_path)
        if (epoch + 1) % 5 == 0:
            end_time = time.time()
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            print('Epoch: {}, time elapse for current epoch: {}'
                  .format(epoch + 1, end_time - start_time))

def normalize(image):
  image = tf.cast(image, tf.float32)
  return image


if __name__ == '__main__':
    dataset = load_celeba("../CelebA/")
    celeba = CelebA(drop_features=[
        'Attractive',
        'Pale_Skin',
        'Blurry',
    ])
    gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    train_split = celeba.split('training', drop_zero=False)
    valid_split = celeba.split('validation', drop_zero=False)
    batch_size = 32
    latent_dim = 64
    generator = Generator()
    discriminator = Discriminator(shape=[32,32,3])
    train_images = normalize(dataset[train_split.index])
    test_images = normalize(dataset[valid_split.index])
    train_attr = train_split.Eyeglasses.to_numpy().reshape(len(train_split), 1)
    test_attr = valid_split.Eyeglasses.to_numpy().reshape(len(valid_split), 1)
    batch_size = 32
    epochs = 100
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                        .shuffle(len(train_split), seed=1).batch(batch_size))
    train_labels = (tf.data.Dataset.from_tensor_slices(train_attr)
                        .shuffle(len(train_split), seed=1).batch(batch_size))

    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(len(valid_split),seed=2).batch(batch_size))
    test_labels = (tf.data.Dataset.from_tensor_slices(test_attr)
                        .shuffle(len(valid_split), seed=2).batch(batch_size))
    date = '4_23/'
    file_path = "cat_test2"
    start_train(epochs, generator, discriminator,
                gen_optimizer, disc_optimizer,
                train_dataset, train_labels,
                test_dataset, test_labels,
                date, file_path)

