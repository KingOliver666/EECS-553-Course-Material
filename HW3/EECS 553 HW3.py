import numpy as np
import pandas as pd

## Import train data and the test data
X_train = np.load('hw2p2_train_x.npy')
X_test = np.load('hw2p2_test_x.npy')

y_train = np.load('hw2p2_train_y.npy')
y_test = np.load('hw2p2_test_y.npy')
##-----------------------------Part(c)
## Get the train data where Y label is 1
index_y_is_1 = np.where(y_train == 1)[0]
X_label1 = X_train[index_y_is_1]

## Get the train data where Y label is 0
index_y_is_0 = np.where(y_train == 0)
X_label0 = X_train[index_y_is_0]

alpha = 1
d = 1000

## Get a list of log(p_1j) for j = 1, ... 1000
n_k1 = np.sum(X_label1)
p_1j = []
for j in range(1000):
    frequency = 0
    for i in range(X_label1.shape[0]):
        frequency = frequency + X_label1[i][j]
    probs = (frequency + alpha)/(n_k1 + alpha * d)
    p_1j.append(np.log(probs))

p_1j[:5]

## Get a list of log(p_0j) for j = 1, ... 1000
n_k0 = np.sum(X_label0)
p_0j = []
for j in range(1000):
    frequency = 0
    for i in range(X_label0.shape[0]):
        frequency = frequency + X_label0[i][j]
    probs = (frequency + alpha)/(n_k0 + alpha * d)
    p_0j.append(np.log(probs))

p_0j[:5]

## Compute the prior pi_0 and pi_0
estimate_pi_1 = np.log(X_label1.shape[0] / X_train.shape[0])
estimate_pi_0 = np.log(X_label0.shape[0] / X_train.shape[0])

print("Estimate of prior pi_0 is:", estimate_pi_0, "Estimate of prior pi_1 is:", estimate_pi_1)



#------------------------------Part(d)
prediction = []
for i in range(X_test.shape[0]):
    y0_value = 0
    y1_value = 0
### Get the value of belong to label 1 or label 0
    for j in range(1000):
        y0_value += X_test[i][j] * p_0j[j]
        y1_value += X_test[i][j] * p_1j[j]
    
    y0_value += estimate_pi_0
    y1_value += estimate_pi_1
### decision rule
    if y0_value > y1_value:
        prediction.append(0)
    else:
        prediction.append(1)

##define a function that to get the accuracy
def accuracy_bayes(X, Y):
    final_result = []
    for i in range(len(X)):
        if X[i] == Y[i]:
            final_result.append(1)
        else:
            final_result.append(0)
    return sum(final_result) / len(final_result)

test_error = 1 - accuracy_bayes(prediction, y_test)

print("The test error is for the naive bayesian classifier is:", test_error)

#------------------------------Part(e)

if_list = [1] * X_test.shape[0]
if_test_error = 1 - accuracy_bayes(if_list, y_test)
print("The test error is for the naive bayesian classifier is:", if_test_error)