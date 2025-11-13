import os
import matplotlib 
import numpy as np

matplotlib.use("Agg") # GPU 비활성화 (추후수정예정)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화 (추후수정예정)

import matplotlib.pyplot as plt # pip install matplotlib
import tensorflow as tf  # pip install tensorflow

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


# Deep Learning Implementation
(x_train, y_train), (x_test, y_test) = mnist.load_data() #load mnist dataset 

#set the structure 784(28X28 image) -> hidden(1024) -> output(10 class)
n_input = 784
n_hidden = 1024
n_output = 10

x_train = x_train.reshape(60000,n_input) #60000 data 28*28 tensor -> 60000 data / 784 feature
x_train = x_train.astype(np.float32)/255 # Nomalize [0~255] to [0,1]

x_test = x_test.reshape(10000, n_input)
x_test = x_test.astype(np.float32)/255

y_train = tf.keras.utils.to_categorical(y_train,n_output)
y_test = tf.keras.utils.to_categorical(y_test,n_output)


mlp = Sequential()

mlp.add(Dense(units=n_hidden, activation='tanh', input_shape=(n_input,), kernel_initializer='random_uniform', bias_initializer='zeros'))

mlp.add(Dense(units=n_output, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))

mlp.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

log = mlp.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test), verbose=2)

result = mlp.evaluate(x_test, y_test, verbose=0)

print("정확률은", result[1] * 100)

plt.plot(log.history['accuracy'])
plt.plot(log.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.grid()
plt.savefig('acc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

plt.plot(log.history['loss'])
plt.plot(log.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.grid()
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("saved: acc_curve.png, loss_curve.png")
