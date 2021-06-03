import tensorflow as tf
from model import CVAE
import numpy as np
import tensorflow_model_optimization as tfmot
from dataset import preprocess_images, divide_dataset
import tempfile
import numpy as np
from tensorflow.linalg import matvec
import time
from tensorflow_addons.image import rotate
import math
def ori_cross_loss(model, r_z, x, d, latent_dim):
    c, s = np.cos(d), np.sin(d)
    latent = latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, -s], [s, c]
    phi_z = rotate_vector(r_z, r_m)
    phi_x = model(phi_z, training=True)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)


def rota_cross_loss(model, z, x, d, r_x, latent_dim):
    c, s = np.cos(d), np.sin(d)
    latent = latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    phi_z = rotate_vector(z, r_m)
    phi_x = model(phi_z, training=True)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=r_x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)


def reconstruction_loss(model, z, X):
    X_pred = model(z, training=True)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return -tf.reduce_mean(logx_z)

def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test




def start_train(epochs, teacher, full_range_set, partial_range_set, date, filePath):
    @tf.function
    def train_step(train_x, degree_set, optimizer):
        for i in range(10, degree_set, 10):
            d = np.radians(i)
            with tf.GradientTape() as tape:
                r_x = rotate(train_x, d)
                mean, logvar = teacher.encode(train_x)
                z = teacher.reparameterize(mean, logvar)
                r_mean, r_logvar = teacher.encode(r_x)
                r_z = teacher.reparameterize(r_mean, r_logvar)
                ori_loss = reconstruction_loss(teacher_for_pruning, z, train_x)
                rota_loss = reconstruction_loss(teacher_for_pruning, r_z, r_x)
                ori_cross_l = ori_cross_loss(teacher_for_pruning, r_z, train_x, d, latent_dim=8)
                rota_cross_l = rota_cross_loss(teacher_for_pruning, z, train_x, d, r_x, latent_dim=8)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
                grads = tape.gradient(total_loss, teacher_for_pruning.trainable_variables)
                optimizer.apply_gradients(zip(grads, teacher_for_pruning.trainable_variables))
    base_model = teacher.decoder

    optimizer = tf.keras.optimizers.Adam()
    log_dir = tempfile.mkdtemp()
    unused_arg = -1


    base_model = teacher.decoder
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=math.ceil(100/batch_size) * epochs)
    }
    teacher_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
      # run pruning callback

    teacher_for_pruning.optimizer = optimizer
    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(teacher_for_pruning)
    log_callback = tfmot.sparsity.keras.PruningSummaries(
        log_dir=log_dir)  # Log sparsity and other metrics in Tensorboard.
    log_callback.set_model(teacher_for_pruning)

    checkpoint_path = "./checkpoints/" + date + filePath
    ckpt = tf.train.Checkpoint(teacher__for_pruning=teacher_for_pruning,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for epoch in range(epochs):
        start_time = time.time()
        log_callback.on_epoch_begin(epoch=unused_arg)
        for train_x in full_range_set:
            step_callback.on_train_batch_begin(batch=unused_arg)
            train_step(teacher, teacher_for_pruning, train_x, 360, optimizer)
        step_callback.on_epoch_end(batch=unused_arg)

        '''
        for train_p in partial_range_set:
            train_step(model, train_p, 180, optimizer) 

        '''
        end_time = time.time()
        loss = tf.keras.metrics.Mean()

        if (epoch + 1) % 100 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
            for i in range(10, 360, 10):
                d = np.radians(i)
                r_x = rotate(train_x, d)
                mean, logvar = teacher.encode(train_x)
                z = teacher.reparameterize(mean, logvar)
                r_mean, r_logvar = teacher.encode(r_x)
                r_z = teacher.reparameterize(r_mean, r_logvar)
                ori_loss = reconstruction_loss(model_for_pruning, z, train_x)
                rota_loss = reconstruction_loss(model_for_pruning, r_z, r_x)
                ori_cross_l = ori_cross_loss(model_for_pruning, r_z, train_x, d, latent_dim=8)
                rota_cross_l = rota_cross_loss(model_for_pruning, z, train_x, d, r_x, latent_dim=8)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
                loss(total_loss)

            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch + 1, elbo, end_time - start_time))


if __name__ == '__main__':
    batch_size = 32
    (mnist_images, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images)

    full_range = mnist_images[np.where(mnist_labels == 7)][:100]
    partial_range = mnist_images[np.where(mnist_labels == 9)][100:200]
    full_range_digit = (tf.data.Dataset.from_tensor_slices(full_range)
                        .batch(batch_size))
    partial_range_digit = (tf.data.Dataset.from_tensor_slices(partial_range)
                           .batch(batch_size))
    model = CVAE(latent_dim=8, beta=6, shape=[28, 28, 1], model='mlp')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore("./checkpoints/5_9/teacher_network/ckpt-8")
    teacher = model

    batch_size = 32
    epochs = 100

    date = '6_1/'
    file_path = 'pruning_teacher/'
    start_train(epochs, model, full_range_digit, partial_range_digit, date, file_path)
