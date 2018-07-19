from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf


def tensors_filter(tensors, filters='', combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(filters, (str, list, tuple)), '`filters` should be a string or a list(tuple) of strings!'
    assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

    if isinstance(filters, str):
        filters = [filters]

    f_tens = []
    for ten in tensors:
        if combine_type == 'or':
            for filt in filters:
                if filt in ten.name:
                    f_tens.append(ten)
                    break
        elif combine_type == 'and':
            all_pass = True
            for filt in filters:
                if filt not in ten.name:
                    all_pass = False
                    break
            if all_pass:
                f_tens.append(ten)
    return f_tens


def get_collection(key, filters='', combine_type='or'):
    tensors = tf.get_collection(key)
    return tensors_filter(tensors, filters, combine_type)

global_variables = partial(get_collection, key=tf.GraphKeys.GLOBAL_VARIABLES)
trainable_variables = partial(get_collection, key=tf.GraphKeys.TRAINABLE_VARIABLES)
update_ops = partial(get_collection, key=tf.GraphKeys.UPDATE_OPS)
