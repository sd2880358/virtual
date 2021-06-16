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




def compute_loss(model, x, y):
    classes = model.cls_dims
    logit_y = model.projection(x)
    loss_t = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels = tf.one_hot(y, classes), logits=logit_y
    ))

    return loss_t


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




def start_train(epochs, model, train_set, test_set, date, filePath):
    @tf.function
    def train_step(model, x, y, optimizer):
            with tf.GradientTape() as tape:
                ori_loss = compute_loss(model, x, y)
                total_loss = ori_loss
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

        for x, y in tf.zip([train_set[0], train_set[1]]):
            train_step(model, x, y, optimizer)

        end_time = time.time()


        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch + 1)%10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            logits = compute_loss(model, test_set[0], test_set[1])
            pred = logits.argmax(-1)
            correct = np.sum(pred == test_labels)
            print('Epoch: {}, Test set loss: {}, accuracy: {}, time elapse for current epoch: {}'
                  .format(epoch+1, logits, correct/float(len(test_labels)), end_time - start_time))

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

    digits_list = [3, 4]
    num_examples_to_generate = 16
    model = CVAE(latent_dim=8, beta=6, shape=[28, 28, 1])
    epochs = 80
    batch_size = 32
    sim_clr = SIM_CLR()
    train_images = (tf.data.Dataset.from_tensor_slices(mnist_images)
            .shuffle(len(mnist_images), seed=1).batch(batch_size))

    train_labels = (tf.data.Dataset.from_tensor_slices(mnist_images)
                    .shuffle(len(mnist_labels), seed=1).batch(batch_size))


    date = '6_16/'
    file_path = 'clr_test1/'
    start_train(epochs, sim_clr, [train_images, train_labels], [test_images, test_labels], date, file_path)