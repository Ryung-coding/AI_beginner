import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split

# perceptron example
sample = np.array([[1,0,0], [1,0,1], [1,1,0],[1,1,1]]) #have a feature
weight = ([-0.5,1.0,1.0])
output = np.sum(sample*weight, axis=1) #calculate the output using z=wx form
print(output)

# perceptron baed on OR data using sklearn Lib
x = [[0,0],[0,1],[1,0],[1,1]] #data
y = [-1,1,1,1] #label or taget

p=Perceptron()
p.fit(x,y)

print("Parm: ",p.coef_,p.intercept_) # -> weight , bias (z=Wx+b form)
print("Predict: ",p.predict(x))
print("Score: ",p.score(x,y)*100,"%")

# Use Train data and Test 
digit = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(digit.data,digit.target,train_size=0.6) # 60% training set / 40% testing set  

p=Perceptron(max_iter=100,eta0=0.001,verbose=0) #model setting value
p.fit(x_train,y_train) # learning about training set

result_prediction = p.predict(x_test)

confusion_mat = np.zeros((10,10)) #check the performance using confusion_matrix
for i in range (len(result_prediction)):
    confusion_mat[result_prediction[i], y_test[i]] +=1
print(confusion_mat)
print("accuracy",np.trace(confusion_mat)*100/confusion_mat.sum())

# -------------------------------------------------> Ground Truth (target) axis=1
# [[62.  0.  0.  0.  0.  0.  1.  0.  0.  0.]    |
#  [ 0. 74.  0.  0.  0.  0.  0.  0.  1.  0.]    |
#  [ 0.  2. 63.  1.  0.  0.  0.  0.  3.  0.]    |
#  [ 0.  0.  2. 85.  0.  2.  0.  1.  4.  3.]    |
#  [ 0.  2.  0.  0. 66.  0.  2.  0.  1.  0.]    |
#  [ 1.  0.  0.  0.  0. 71.  0.  0.  0.  1.]    |
#  [ 0.  2.  0.  0.  0.  1. 70.  0.  1.  0.]    |
#  [ 0.  0.  0.  0.  0.  0.  0. 67.  1.  0.]    V  Preditied class axis=0
#  [ 0.  0.  1.  0.  0.  0.  0.  0. 57.  0.]
#  [ 0.  0.  0.  1.  1.  0.  0.  1.  1. 67.]]

# have a off-digonal term => can't predit the targer!!

