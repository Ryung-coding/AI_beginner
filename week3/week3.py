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
mlp.fit(x_train,y_train)

result_prediction = mlp.predict(x_test)

confusion_mat = np.zeros((10,10)) #check the performance using confusion_matrix
for i in range (len(result_prediction)):
    confusion_mat[result_prediction[i], y_test[i]] +=1
print(confusion_mat)
print("accuracy",np.trace(confusion_mat)*100/confusion_mat.sum())

