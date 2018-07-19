from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob as glob

import pylib
import tensorflow as tf
import tflib as tl


def get_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        # dataset
        pylib.mkdir('./data/mnist')
        dataset = tl.Mnist(data_dir='./data/mnist', batch_size=batch_size, repeat=1)

        # get next func
        def get_next():
            return dataset.get_next()['img']

        dataset.get_next_ = get_next

        # shape
        img_shape = [28, 28, 1]

    elif dataset_name == 'celeba':
        # dataset
        def _map_func(img):
            crop_size = 108
            re_size = 64
            img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
            img = tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            return img

        paths = glob.glob('./data/celeba/img_align_celeba/*.jpg')
        dataset = tl.DiskImageData(img_paths=paths, batch_size=batch_size, repeat=1, map_func=_map_func)

        # get next func
        dataset.get_next_ = dataset.get_next

        # shape
        img_shape = [64, 64, 3]

    return dataset, img_shape
