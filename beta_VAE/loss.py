import tensorflow as tf
import numpy as np
from tensorflow.linalg import matvec

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

def reconstruction_loss(model, X):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return -tf.reduce_mean(logx_z)

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

def kl_divergence(mean, logvar):
    summand = tf.math.square(mean) + tf.math.exp(logvar) - logvar  - 1
    return (0.5 * tf.reduce_sum(summand, [1]))

def compute_loss(model, x):
    beta = model.beta
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))
    log_qz, logq_z_product = estimate_entropies(z, mean, logvar)
    tc = tf.reduce_mean(log_qz - logq_z_product)
    kl_loss = tf.reduce_mean(kl_divergence(mean, logvar))

    return tf.reduce_mean(logx_z + kl_loss + (beta-1) * tc)

def gaussian_log_density(samples, mean, logvar):
    pi = tf.constant(np.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-logvar)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)
