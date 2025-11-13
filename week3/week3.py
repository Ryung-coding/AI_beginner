import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#use MLP using MNIST dataset
digit=datasets.load_digits()

plt.figure(figsize=(5,5))
plt.imshow(digit.images[0], cmap=plt.cm.gray_r, interpolation='nearest') #digit.images[i] => i-th gray image
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(digit.data,digit.target,train_size=0.6) 

# input data(8X8 image) -> hidden_layer_sizes (expend) -> output1 -> last_layer_size = output_class_number / in out case = 10
#          64                       100                     100                     10                        =>64*100*100*10 
mlp=MLPClassifier(hidden_layer_sizes=(100), learning_rate_init=0.001,batch_size=32, max_iter=300, solver='sgd', verbose=True)
hidden_layer_sizes=(100)

# hidden_layer_sizes=(100)  ==> 은닉층 1개, 뉴런 100개
# learning_rate_init=0.001  ==> 초기 학습률(η). solver='sgd'나 'adam'에서 사용. 너무 크면 발산, 너무 작으면 수렴 느림.
# batch_size=32             ==> 한 번의 가중치 업데이트에 쓰는 샘플 수. solver='sgd'/'adam'에서만 의미 있음.
# max_iter=300              ==> 에폭 수(전체 데이터셋을 1회 도는 횟수). 조기 종료가 없으면 최대 300에폭 학습.
# solver='sgd'              => 'sgd': 확률적 경사하강법(+옵션으로 모멘텀/학습률 스케줄).
                                # Note* 'adam': 보통 기본값 | 'lbfgs': 소규모 데이터에서 빠르게 수렴
# verbose=True              ==> loss/상태를 로그로 출력.

mlp.fit(x_train,y_train)

result_prediction = mlp.predict(x_test)

confusion_mat = np.zeros((10,10)) #check the performance using confusion_matrix
for i in range (len(result_prediction)):
    confusion_mat[result_prediction[i], y_test[i]] +=1
print(confusion_mat)
print("accuracy",np.trace(confusion_mat)*100/confusion_mat.sum())

