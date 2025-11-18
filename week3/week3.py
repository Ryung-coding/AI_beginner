import matplotlib
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf # pip install --upgrade "tensorflow[and-cuda]"
print("GPUs:", tf.config.list_physical_devices("GPU")) #chcek
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype(np.float32) / 255.0
x_test  = x_test.reshape(10000, 28, 28, 1).astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

cnn = Sequential()

cnn.add(Conv2D(6, (5,5), activation='tanh', padding='same',input_shape=(28,28,1),kernel_initializer='random_uniform', bias_initializer='zeros'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Conv2D(16, (5,5), activation='tanh', padding='same',kernel_initializer='random_uniform', bias_initializer='zeros'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Conv2D(120, (5,5), activation='tanh', padding='valid',kernel_initializer='random_uniform', bias_initializer='zeros'))
cnn.add(Flatten())

cnn.add(Dense(84, activation='tanh',kernel_initializer='random_uniform', bias_initializer='zeros'))
cnn.add(Dense(10, activation='tanh',kernel_initializer='random_uniform', bias_initializer='zeros'))

cnn.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
log = cnn.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test), verbose=2)
result = cnn.evaluate(x_test, y_test, verbose=0)

print("정확률은", result[1] * 100)

plt.plot(log.history['accuracy'])
plt.plot(log.history['val_accuracy'])
plt.title('Model accuracy (CNN)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.grid()
plt.savefig('cnn_acc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

plt.plot(log.history['loss'])
plt.plot(log.history['val_loss'])
plt.title('Model loss (CNN)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.grid()
plt.savefig('cnn_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("saved: cnn_acc_curve.png, cnn_loss_curve.png")     
