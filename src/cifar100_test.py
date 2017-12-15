import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from utils.load_data_cifar import load_cifar_100

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='../dataset/cifar-100-python')
parser.add_argument('--batch_size', default=64)
FLAGS = parser.parse_args()


# define model
def model_dnn(input_dim, hidden_neurons=[256, 64], output_dim=10):
    model = Sequential()
    model.add(Dense(units=hidden_neurons[0], activation='relu', input_dim=input_dim))
    model.add(Dense(units=hidden_neurons[1], activation='relu'))
    model.add(Dense(units=output_dim, activation='softmax'))
    return model


# load cifar100
(x_train, y_train), (x_test, y_test) = load_cifar_100(FLAGS.dataset_path, label_mode='fine', flat=True)


# build model
DNN = model_dnn(3072, [256, 128], 100)
print DNN.summary()


# define optimizer
DNN.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# training
DNN.fit(x_train, to_categorical(y_train, num_classes=100), epochs=200, batch_size=32)
