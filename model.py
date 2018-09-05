from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl


# ==============================================================================
# =                                    alias                                   =
# ==============================================================================

conv = partial(tl.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected_v2, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
# batch_norm = partial(slim.batch_norm, scale=True)
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
layer_norm = slim.layer_norm
instance_norm = slim.instance_norm
# spectral_norm = tl.spectral_normalization
spectral_norm = partial(tl.spectral_normalization, updates_collections=None)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

def _get_norm_fn(norm_name, is_training):
    if norm_name == 'none':
        norm = None
    elif norm_name == 'batch_norm':
        norm = partial(batch_norm, is_training=is_training)
    elif norm_name == 'instance_norm':
        norm = instance_norm
    elif norm_name == 'layer_norm':
        norm = layer_norm
    return norm


def _get_weight_norm_fn(weights_norm_name, is_training):
    if weights_norm_name == 'none':
        weights_norm = None
    elif weights_norm_name == 'spectral_norm':
        weights_norm = partial(spectral_norm, is_training=is_training)
    return weights_norm


def G_conv_mnist(z, dim=64, is_training=True):
    norm = _get_norm_fn('batch_norm', is_training)
    fc_norm_relu = partial(fc, normalizer_fn=norm, activation_fn=relu)
    dconv_norm_relu = partial(dconv, normalizer_fn=norm, activation_fn=relu)

    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        y = fc_norm_relu(z, 1024)
        y = fc_norm_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_norm_relu(y, dim * 2, 4, 2)
        x = tf.tanh(dconv(y, 1, 4, 2))
        return x


def D_conv_mnist(x, dim=64, is_training=True, norm_name='batch_norm', weights_norm_name='none'):
    norm = _get_norm_fn(norm_name, is_training)
    weights_nome = _get_weight_norm_fn(weights_norm_name, is_training)
    fc_norm_lrelu = partial(fc, normalizer_fn=norm, weights_normalizer_fn=weights_nome, activation_fn=lrelu)
    conv_norm_lrelu = partial(conv, normalizer_fn=norm, weights_normalizer_fn=weights_nome, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = conv_norm_lrelu(x, 1, 4, 2)
        y = conv_norm_lrelu(y, dim, 4, 2)
        y = fc_norm_lrelu(y, 1024)  # fc_norm doesn't work with instance normalization
        logit = fc(y, 1, weights_normalizer_fn=weights_nome)
        # logit = fc(y, 1)
        return logit


def G_conv_64(z, dim=64, is_training=True):
    norm = _get_norm_fn('batch_norm', is_training)
    fc_norm_relu = partial(fc, normalizer_fn=norm, activation_fn=relu)
    dconv_norm_relu = partial(dconv, normalizer_fn=norm, activation_fn=relu)

    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        y = fc_norm_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = dconv_norm_relu(y, dim * 4, 4, 2)
        y = dconv_norm_relu(y, dim * 2, 4, 2)
        y = dconv_norm_relu(y, dim * 1, 4, 2)
        x = tf.tanh(dconv(y, 3, 4, 2))
        return x


def D_conv_64(x, dim=64, is_training=True, norm_name='batch_norm', weights_norm_name='none'):
    norm = _get_norm_fn(norm_name, is_training)
    weights_nome = _get_weight_norm_fn(weights_norm_name, is_training)
    conv_norm_lrelu = partial(conv, normalizer_fn=norm, weights_normalizer_fn=weights_nome, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = conv_norm_lrelu(x, dim, 4, 2)
        y = conv_norm_lrelu(y, dim * 2, 4, 2)
        y = conv_norm_lrelu(y, dim * 4, 4, 2)
        y = conv_norm_lrelu(y, dim * 8, 4, 2)
        logit = fc(y, 1, weights_normalizer_fn=weights_nome)
        # logit = fc(y, 1)
        return logit


def get_models(model_name):
    if model_name == 'conv_mnist':
        return G_conv_mnist, D_conv_mnist
    elif model_name == 'conv_64':
        return G_conv_64, D_conv_64


# ==============================================================================
# =                                loss function                               =
# ==============================================================================

def get_loss_fn(mode):
    if mode == 'gan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(r_logit), r_logit)
            f_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(f_logit), f_logit)
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(f_logit), f_logit)
            return f_loss

    elif mode == 'lsgan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = tf.losses.mean_squared_error(tf.ones_like(r_logit), r_logit)
            f_loss = tf.losses.mean_squared_error(tf.zeros_like(f_logit), f_logit)
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = tf.losses.mean_squared_error(tf.ones_like(f_logit), f_logit)
            return f_loss

    elif mode == 'wgan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = - tf.reduce_mean(r_logit)
            f_loss = tf.reduce_mean(f_logit)
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = - tf.reduce_mean(f_logit)
            return f_loss

    elif mode == 'hinge':
        def d_loss_fn(r_logit, f_logit):
            r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
            f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            # f_loss = tf.reduce_mean(tf.maximum(1 - f_logit, 0))
            f_loss = tf.reduce_mean(- f_logit)
            return f_loss

    return d_loss_fn, g_loss_fn


# ==============================================================================
# =                                   others                                   =
# ==============================================================================

def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            with tf.name_scope('interpolate'):
                if b is None:   # interpolation in DRAGAN
                    beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                    _, variance = tf.nn.moments(a, range(a.shape.ndims))
                    b = a + 0.5 * tf.sqrt(variance) * beta
                shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
                alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.get_shape().as_list())
                return inter

        with tf.name_scope('gradient_penalty'):
            x = _interpolate(real, fake)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = tf.gradients(pred, x)[0]
            norm = tf.norm(slim.flatten(grad), axis=1)
            gp = tf.reduce_mean((norm - 1.)**2)
            return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=tf.float32)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)

    return gp
