from model import CVAE
import tensorflow as tf
import numpy as np


def reconstruction_loss(model, X):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return -tf.reduce_mean(logx_z)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, teacher_1, teacher_2, X):
    mean, logvar = model.encode(X)
    t1_mean, t1_logvar = teacher_1.encode(X)
    t1_z = teacher_1.reparameterize(t1_mean, t1_logvar)
    t2_mean, t2_logvar = teacher_2.encode(X)
    t2_z = teacher_2.reparameterize(t2_mean, t2_logvar)
    z = model.reparameterize(mean, logvar)
    logpz_x = log_normal_pdf(z, mean, logvar)
    log_t1 = log_normal_pdf(t1_z, 0, 0)
    log_t2 = log_normal_pdf(t2_z, 0, 0)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=X)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return -tf.reduce_mean(logpx_z + 0.5 * (log_t1 + log_t2 - 2 * logpz_x ))


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
    degree = np.radians(random.randint(30, 90))
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
        r_sample = rotate(test_sample, degree)
    generate_and_save_images(model, 0, test_sample, file_path)
    generate_and_save_images(model, 0, r_sample, "rotate_image")
    display.clear_output(wait=False)
    in_range_socres = []
    mean, logvar = model.encode(test_images)
    r_m = np.identity(model.latent_dim)
    z = model.reparameterize(mean, logvar)
    for i in range(0, 100, 10):
        theta = np.radians(i)
        scores = compute_mnist_score(model, classifier, z, theta, r_m)
        in_range_socres.append(scores)
    score = np.mean(in_range_socres)
    iteration = 0
    for epoch in range(epochs):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
            iteration += 1
        end_time = time.time()
        loss = tf.keras.metrics.Mean()
        epochs += 1
        in_range_socres = []
        mean, logvar = model.encode(test_images)
        r_m = np.identity(model.latent_dim)
        z = model.reparameterize(mean, logvar)
        for i in range(0, 100, 10):
            theta = np.radians(i)
            scores = compute_mnist_score(model, classifier, z, theta, r_m)
            in_range_socres.append(scores)
        score = np.mean(in_range_socres)
        #generate_and_save_images(model, epochs, test_sample, file_path)
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch + 1)%1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            compute_and_save_mnist_score(model, classifier, iteration, file_path)
            for test_x in test_dataset:
                d = np.radians(random.randint(30, 90))
                r_x = rotate(test_x, d)
                total_loss = rota_cross_loss(model, test_x, d) \
                             + ori_cross_loss(model, test_x, d) \
                             + compute_loss(model, test_x) \
                             + reconstruction_loss(model, r_x)
                loss(total_loss)
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epochs, elbo, end_time - start_time))
            print('The current score is {}', score)

if __name__ == '__main__':
