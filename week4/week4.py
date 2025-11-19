import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import CosineDecay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 데이터세트 설정
num_train_samples = 55000
batch_size = 100
input_dim = 784
num_classes = 10

# 모델 설정
hidden_dim = 2000

# 훈련 설정
epochs = 60
lr = 1e-2
momentum = 0.9

# Fashion MNIST 데이터세트 다운로드
(train_image, train_label), (test_image, test_label) = fashion_mnist.load_data()

# Fahsion MNIST 데이터세트 확인
print('훈련 이미지: ', train_image.shape)
print('테스트 이미지: ', test_image.shape)
print('훈련 라벨: ', train_label.shape)
print('테스트 라벨: ', test_label.shape)

# 훈련세트를 훈련세트 / 검증세트로 분할
valid_image = train_image[num_train_samples:, :, :]
valid_label = train_label[num_train_samples:]
train_image = train_image[:num_train_samples, :, :]
train_label = train_label[:num_train_samples]

# 데이터처리 파이프라인 1: tensorflow Dataset 클래스 생성 
train_dataset = Dataset.from_tensor_slices((train_image, train_label))
valid_dataset = Dataset.from_tensor_slices((valid_image, valid_label))
test_dataset = Dataset.from_tensor_slices((test_image, test_label))


# 데이터처리 파이프라인 2: 전처리함수 및 미니배치 그룹핑 
def preprocess(image, label):
    """ 이미지, 라벨 전처리 과정
        1) 이미지 자료형 변환: 'uint8' -> 'float32'
        2) 이미지 값 범위 정규화: 0 ~ 255 -> 0 ~ 1
        3) 이미지 크기 변환: (28, 28) -> (784,)
        4) 라벨을 원핫벡터로 변환
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, (input_dim,))
    label = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    return image, label

train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.batch(batch_size)

valid_dataset = valid_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size)

test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)

# 모델 정의
inputs = Input(shape=(input_dim,), name='input')
x = Dense(hidden_dim, activation='relu', name='hidden1')(inputs)
x = Dense(hidden_dim, activation='relu', name='hidden2')(x)
x = Dense(num_classes, activation="softmax", name='output')(x)
model = Model(inputs, x, name='mlp')

model.summary()

# 평가지표 정의
metric = CategoricalAccuracy()

# 손실함수 정의
loss = CategoricalCrossentropy()

# 학습률 스케쥴러 정의
lr_schedule = CosineDecay(lr, len(train_dataset) * epochs)

# 최적화 알고리즘 정의
optimizer = SGD(lr_schedule, momentum=momentum)

# 모델 컴파일
model.compile(optimizer=optimizer, 
              loss=loss, 
              metrics=[metric])

# 모델 훈련
history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset, verbose=1)

train_result = model.evaluate(train_dataset, verbose=0)   # [loss, acc]
valid_result = model.evaluate(valid_dataset, verbose=0)   # [loss, acc]
test_result  = model.evaluate(test_dataset,  verbose=0)   # [loss, acc]

print(f"Train  accuracy: {train_result[1] * 100:.2f}%")
print(f"Valid  accuracy: {valid_result[1] * 100:.2f}%")
print(f"Test   accuracy: {test_result[1] * 100:.2f}%")

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy (MLP))')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.grid()
plt.savefig('60_MLP2_acc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss (MLP))')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.grid()
plt.savefig('60_MLP2_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("saved: 60_MLP2_acc_curve.png, 60_MLP2_loss_curve.png")