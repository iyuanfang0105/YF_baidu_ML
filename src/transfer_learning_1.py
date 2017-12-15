import argparse
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.contrib.keras.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.applications import inception_v3, imagenet_utils
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

from utils.image_io import create_image_dataset

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='../dataset/flowers/flower_photos')
parser.add_argument('--training_percentage', type=float, default=0.8)
parser.add_argument('--testing_percentage', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--input_image_size', type=int, default=299)
FLAGS, unparsed = parser.parse_known_args()

# load pretrained model
model_inception_v3 = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
                                              input_shape=None, pooling=None)
# add a global spatial average pooling layer
x = model_inception_v3.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(5, activation='softmax')(x)
# the model for transform learning
model = Model(inputs=model_inception_v3.input, outputs=predictions)
print model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model_inception_v3.layers:
    layer.trainable = False

# load dataset
# flowers_dataset = create_image_dataset(FLAGS.image_dir, training_percentage=FLAGS.training_percentage,
#                                        testing_percentage=FLAGS.testing_percentage)
image_data_generator = ImageDataGenerator()
train_data = image_data_generator.flow_from_directory('/home/meizu/WORK/code/YF_baidu_ML/dataset/flowers/flower_photos/train',
                                             target_size=(FLAGS.input_image_size, FLAGS.input_image_size),
                                             batch_size=FLAGS.batch_size,
                                             class_mode='categorical')
test_data = image_data_generator.flow_from_directory('/home/meizu/WORK/code/YF_baidu_ML/dataset/flowers/flower_photos/test',
                                             target_size=(FLAGS.input_image_size, FLAGS.input_image_size),
                                             batch_size=FLAGS.batch_size,
                                             class_mode='categorical')

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

train_data_count = len(train_data.filenames)
test_data_count = len(test_data.filenames)
model.fit_generator(train_data,
                    steps_per_epoch=(train_data_count // FLAGS.batch_size + 1),
                    epochs=1,
                    verbose=1,
                    validation_data=test_data,
                    validation_steps=(test_data_count // FLAGS.batch_size + 1))

# a = model_inception_v3.predict_generator(g, len(g.filenames))
# print ''
# a = model_inception_v3.predict_generator(g, steps=len(g.filenames))
#
# # extract features
# for label_name, label_lists in flowers_dataset.items():
#     for category in ['training', 'testing', 'validation']:
#         category_list = label_lists[category]
#         for index, image_path in enumerate(category_list):
#             if not gfile.Exists(image_path):
#                 tf.logging.fatal('File does not exist %s', image_path)
#             img = misc.imresize(misc.imread(image_path), [299, 299]).astype(np.float32)
#             img = imagenet_utils.preprocess_input(np.expand_dims(img, 0))
#             plt.imshow(img[0])
#             plt.show()
#             a = model_inception_v3.predict(img)
