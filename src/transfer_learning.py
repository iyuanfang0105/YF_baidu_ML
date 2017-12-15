from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf


from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from utils.image_io import create_image_lists, add_input_distortions


def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it.
    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.
    Args:
      architecture: Name of a model architecture.
    Returns:
      Dictionary of information about the model, or None if the name isn't
      recognized
    Raises:
      ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()
    is_quantized = False
    # pylint: disable=line-too-long
    if architecture == 'inception_v3':
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'quantize_layer': is_quantized,
    }


def create_model_graph(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Args:
      model_info: Dictionary containing information about the model architecture.
    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        tf.logging.info('Model path: ' + model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


def main(_):
    # creat model info
    tf.logging.info("************* Creating model info *************")
    model_info = create_model_info(FLAGS.architecture)

    # rebuild graph according pretrained model
    tf.logging.info("************* Rebuild graph according pretrained model *************")
    graph, bottleneck_tensor, resized_image_tensor = create_model_graph(model_info)

    # load dataset
    tf.logging.info("************* Load dataset **************")
    dataset = create_image_lists(FLAGS.image_dir, training_percentage=0.7, testing_percentage=0.2)
    if len(dataset.keys()) <= 1:
        tf.logging.error('multiple classes are needed for classification ' + FLAGS.image_dir)
        return -1

    # define an operation of distorting images
    image_tensor, distorted_image_tensor = add_input_distortions(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness, model_info['input_width'],
        model_info['input_height'], model_info['input_depth'],
        model_info['input_mean'], model_info['input_std'])

    with tf.Session(graph=graph) as sess:
        image_data = gfile.FastGFile(dataset, 'rb').read()
        # First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='../pretrained_models')
    parser.add_argument('--architecture', type=str, default='inception_v3')
    parser.add_argument('--image_dir', type=str, default='../dataset/flowers/flower_photos')
    parser.add_argument('--training_percentage', type=float, default=0.8)
    parser.add_argument('--testing_percentage', type=float, default=0.2)
    parser.add_argument('--flip_left_right', type=bool, default=False,)
    parser.add_argument('--random_crop', type=int, default=0)
    parser.add_argument('--random_scale', type=int, default=0)
    parser.add_argument('--random_brightness', type=int, default=0)


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    print("")