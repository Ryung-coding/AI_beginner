import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


arr = np.array([1,2,3])
# print(f"=> {type(arr)} {arr.ndim} {arr.shape}")

arr_zero = np.zeros((2,3))
arr_one = np.ones((2,3))
# print(arr_zero, arr_one)

arr_def = np.arange(3)
arr_def_prime = np.arange(start=1, stop=4)
# print(arr_def, arr_def_prime)

arr_2dim = np.array([[1,2,3],[4,5,6]])
arr_3dim = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]], [[13,14,15],[-1,-1,-1]]])
# print(arr_3dim, arr_3dim[1,0,1])

arr_sort = np.array([[5,2,7],[4,3,6]])
#print(np.sort(arr_sort,axis=1))

#print(arr_2dim + arr_sort)
#print(arr_2dim * arr_sort)
#print(np.dot(np.array([1,2,3]), np.array([1,2,3])))

#igea molgga.. jom mianheajinea..

#Ex1
a = np.array([1,0,3])
b = np.array([2,1,2])
print(2*a+3*b)

#Ex2
x = np.array([1,2,3,-1])
y = np.array([0,1,2,+1])
z = np.array([2023,1,-1,+1])
print(np.dot(x,y), np.dot(y,z))

#Ex3
arr_R3 = np.array([[11,10,3,4],[7,1,2,9],[6,8,5,12  ]])
print(arr_R3)
buf = (np.sort(arr_R3.reshape(1,-1))).reshape(3,4)
print(arr_R3.sum(axis=0), arr_R3.mean(axis=1), buf)
