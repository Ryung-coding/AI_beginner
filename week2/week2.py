import numpy as np
import pandas as pd
from sklearn import datasets, svm                       # pip install scikit-learn

# https://www.kaggle.com/datasets/heptapod/titanic      <- dataset OR Use under the code!!
# import kagglehub                                      <- pip install kagglehub
# path = kagglehub.dataset_download("heptapod/titanic")
# print("Path to dataset files:", path)                 # mv train_and_test2.csv new_folder_path 
                                                        #for Me : /home/ryung/Desktop/AI_example/week2/

# data = pd.read_csv("train_and_test2.csv")             # about Titanic dataSet
# print(data.head(5))

#apply SVM(Support vector machine) Model 
d = datasets.load_iris() # about Iris(꽃) dataSet
# print(d.DESCR)
# for i in range(0,len(d.data)):
#     print(i+1, d.data[i], d.target[i]) # --> [6.7 3.3 5.7 2.5] 2 , When you look at the data as vector, find out the trend between data and targer

s=svm.SVC(gamma=0.1,C=10)
s.fit(d.data,d.target) #training 

new_data = [[6.4, 3.2, 6.0, 2.5],[7.1, 3.1, 4.7, 1.35]]

res=s.predict(new_data)
print("prediction result => ", res) # this code predict the targer 

predict_all = s.predict(d.data)
confusion_mat = np.zeros((3,3)) # To use a confusion_mat, we can ckeck the perfomance about trend(data&target)

for i in range(0, len(predict_all)):
    confusion_mat[predict_all[i], d.target[i]]+=1
print(confusion_mat)
