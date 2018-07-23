from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback

import data
import imlib as im
import model
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

# argument
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=100)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.0002, help='learning rate of d')
parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.0002, help='learning rate of g')
parser.add_argument('--n_d', dest='n_d', type=int, default=1)
parser.add_argument('--n_g', dest='n_g', type=int, default=1)
parser.add_argument('--n_d_pre', dest='n_d_pre', type=int, default=0)
parser.add_argument('--optimizer', dest='optimizer', default='adam', choices=['adam', 'rmsprop'])

parser.add_argument('--z_dim', dest='z_dim', type=int, default=100, help='dimension of latent')
parser.add_argument('--loss_mode', dest='loss_mode', default='gan', choices=['gan', 'lsgan', 'wgan', 'hinge'])
parser.add_argument('--gp_mode', dest='gp_mode', default='none', choices=['none', 'dragan', 'wgan-gp'], help='type of gradient penalty')
parser.add_argument('--gp_coef', dest='gp_coef', type=float, default=10.0, help='coefficient of gradient penalty')
parser.add_argument('--norm', dest='norm', default='batch_norm', choices=['batch_norm', 'instance_norm', 'layer_norm', 'none'])
parser.add_argument('--weights_norm', dest='weights_norm', default='none', choices=['none', 'spectral_norm', 'weight_clip'])
parser.add_argument('--vgan', dest='vgan', action='store_true', help='use vgan')
parser.add_argument('--vgan_coef', dest='vgan_coef', type=float, default=5.0, help='coefficient of vgan regularization')

parser.add_argument('--model', dest='model_name', default='conv_mnist', choices=['conv_mnist', 'conv_64'])
parser.add_argument('--dataset', dest='dataset_name', default='mnist', choices=['mnist', 'celeba'])
parser.add_argument('--experiment_name', dest='experiment_name', default='default')

args = parser.parse_args()

epoch = args.epoch
batch_size = args.batch_size
lr_d = args.lr_d
lr_g = args.lr_g
n_d = args.n_d
n_g = args.n_g
n_d_pre = args.n_d_pre
optimizer = args.optimizer

z_dim = args.z_dim
loss_mode = args.loss_mode
gp_mode = args.gp_mode
gp_coef = args.gp_coef
norm = args.norm
weights_norm = args.weights_norm
vgan = args.vgan
vgan_coef = args.vgan_coef

model_name = args.model_name

dataset_name = args.dataset_name
experiment_name = args.experiment_name

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# dataset
dataset, img_shape = data.get_dataset(dataset_name, batch_size)


# ==============================================================================
# =                                    graph                                   =
# ==============================================================================

# models
G, D = model.get_models(model_name)
D = partial(D, norm_name=norm, weights_norm_name='spectral_norm' if weights_norm == 'spectral_norm' else 'none')

# loss func
d_loss_fn, g_loss_fn = model.get_losses(loss_mode)

# inputs
real = tf.placeholder(tf.float32, [None] + img_shape)
z = tf.placeholder(tf.float32, [None, z_dim])

# generate
fake = G(z)

# dicriminate
r_logit = D(real)
f_logit = D(fake)

# d loss
d_r_loss, d_f_loss = d_loss_fn(r_logit, f_logit)
gp = model.gradient_penalty(D, real, fake, gp_mode)
d_loss = d_r_loss + d_f_loss + gp * gp_coef

# g loss
g_loss = g_loss_fn(f_logit)
if vgan:
    g_loss += tf.losses.mean_squared_error(tf.stop_gradient(fake), fake) * vgan_coef

# otpims
if optimizer == 'adam':
    optim = partial(tf.train.AdamOptimizer, beta1=0.5)
elif optimizer == 'rmsprop':
    optim = tf.train.RMSPropOptimizer

with tf.control_dependencies(tl.update_ops(includes='D')):
    d_step = optim(learning_rate=lr_d).minimize(d_loss, var_list=tl.trainable_variables(includes='D'))
    if weights_norm == 'weight_clip':
        with tf.control_dependencies([d_step]):
            d_step = tf.group(*(tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in tl.trainable_variables(includes='D')))
with tf.control_dependencies(tl.update_ops(includes='G')):
    g_step = optim(learning_rate=lr_g).minimize(g_loss, var_list=tl.trainable_variables(includes='G'))

# summaries
d_summary = tl.summary({d_loss: 'd_loss',
                        d_r_loss: 'd_r_loss',
                        d_f_loss: 'd_f_loss',
                        gp: 'gp'}, scope='D')
g_summary = tl.summary({g_loss: 'g_loss'}, scope='G')

# sample
z_sample = tf.placeholder(tf.float32, [None, z_dim])
f_sample = G(z_sample, is_training=False)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# epoch counter
ep_cnt, update_cnt = tl.counter(start=1)

# session
sess = tl.session()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

# train
try:
    z_ipt_sample = np.random.normal(size=[100, z_dim])

    it = 0
    it_per_epoch = len(dataset) // (batch_size * n_d)
    for ep in range(sess.run(ep_cnt), epoch + 1):
        sess.run(update_cnt)

        dataset.reset()
        for i in range(it_per_epoch):
            it += 1

            # train D
            if n_d_pre > 0 and it <= 25:
                n_d_ = n_d_pre
            else:
                n_d_ = n_d
            for _ in range(n_d_):
                # batch data
                real_ipt = dataset.get_next_()
                z_ipt = np.random.normal(size=[batch_size, z_dim])

                d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
                summary_writer.add_summary(d_summary_opt, it)

            # train G
            for _ in range(n_g):
                # batch data
                z_ipt = np.random.normal(size=[batch_size, z_dim])

                g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
                summary_writer.add_summary(g_summary_opt, it)

            # display
            if it % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (ep, i + 1, it_per_epoch))

            # sample
            if it % 1000 == 0:
                f_sample_opt = sess.run(f_sample, feed_dict={z_sample: z_ipt_sample}).squeeze()

                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)
                im.imwrite(im.immerge(f_sample_opt), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, ep, i + 1, it_per_epoch))

        save_path = saver.save(sess, '%s/Epoch_%d.ckpt' % (ckpt_dir, ep))
        print('Model is saved in file: %s' % save_path)
except:
    traceback.print_exc()
finally:
    sess.close()
