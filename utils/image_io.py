from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import random
import math
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.keras.python.keras.applications.inception_v3 import preprocess_input

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_dataset(image_dir, training_percentage=0.7, testing_percentage=0.2):
    """Builds a list of training images from the file system.
    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.
    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.
    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in " + dir_name)
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        # tf.logging.info(dir_name + " contains " + str(len(file_list)) + " images")
        # training_images = []
        # testing_images = []
        # validation_images = []
        # for file_name in file_list:
            # base_name = os.path.basename(file_name)
            # # We want to ignore anything after '_nohash_' in the file name when
            # # deciding which set to put an image in, the data set creator has a way of
            # # grouping photos that are close variations of each other. For example
            # # this is used in the plant disease data set to group multiple pictures of
            # # the same leaf.
            # hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # # This looks a bit magical, but we need to decide whether this file should
            # # go into the training, testing, or validation sets, and we want to keep
            # # existing files in the same set even if more files are subsequently
            # # added.
            # # To do that, we need a stable way of deciding based on just the file name
            # # itself, so we do a hash of that and then use that to generate a
            # # probability value that we use to assign it.
            # hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            # percentage_hash = ((int(hash_name_hashed, 16) %
            #                     (MAX_NUM_IMAGES_PER_CLASS + 1)) *
            #                    (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            # if percentage_hash < validation_percentage:
            #     validation_images.append(base_name)
            # elif percentage_hash < (testing_percentage + validation_percentage):
            #     testing_images.append(base_name)
            # else:
            #     training_images.append(base_name)
        random.shuffle(file_list)
        training_images = file_list[:int(math.floor(len(file_list)*training_percentage))]
        testing_images = file_list[int(math.floor(len(file_list)*training_percentage)):int(math.floor(len(file_list)*(training_percentage+testing_percentage)))]
        validation_images = file_list[int(math.floor(len(file_list) * (training_percentage + testing_percentage))):]
        tf.logging.info(dir_name + " contains " + str(len(file_list)) + " images, " + " training:" + str(len(training_images)) + " testing:" + str(len(testing_images)) + " validation:" + str(len(validation_images)))
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }

    return result


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
    """Creates the operations to apply the specified distortions.
    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.
    Cropping
    ~~~~~~~~
    Cropping is done by placing a bounding box at a random position in the full
    image. The cropping parameter controls the size of that box relative to the
    input image. If it's zero, then the box is the same size as the input and no
    cropping is performed. If the value is 50%, then the crop box will be half the
    width and height of the input. In a diagram it looks like this:
    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+
    Scaling
    ~~~~~~~
    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example if
    the scale percentage is zero, then the bounding box is the same size as the
    input and no scaling is applied. If it's 50%, then the bounding box will be in
    a random range between half the width and height and full size.
    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin used around the
      crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply the pixel values by.
      graph.
      input_width: Horizontal size of expected input image to model.
      input_height: Vertical size of expected input image to model.
      input_depth: How many channels the expected input image should have.
      input_mean: Pixel value that should be zero in the image for the graph.
      input_std: How much to divide the pixel values by before recognition.
    Returns:
      The jpeg input layer and the distorted result tensor.
    """

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../dataset/flowers/flower_photos')
    parser.add_argument('--training_percentage', type=float, default=0.8)
    parser.add_argument('--testing_percentage', type=float, default=0.2)
    FLAGS, unparsed = parser.parse_known_args()
    flowers_dataset = create_image_dataset(FLAGS.image_dir, training_percentage=FLAGS.training_percentage, testing_percentage=FLAGS.testing_percentage)

    for label_name, label_lists in flowers_dataset.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, image_path in enumerate(category_list):
                if not gfile.Exists(image_path):
                    tf.logging.fatal('File does not exist %s', image_path)
                # image_str = gfile.FastGFile(image_path, 'rb').read()

                img = preprocess_input(misc.imresize(misc.imread(image_path), [229, 229]).astype(np.float32))
                plt.imshow(img)
                plt.show()
                print()
    # get_image_path(flowers_dataset, )
    print("")