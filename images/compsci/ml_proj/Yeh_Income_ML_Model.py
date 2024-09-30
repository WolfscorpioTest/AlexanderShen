# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:10:50 2021
Logistic Regression Model for Predicting Annual Income

"""

# In[1]:

# Open and load the data using numpy arrays.
import numpy as np

np.random.seed(0)

x_train_fpath = 'C:/Users/Joy/Documents/Joy/Intercession 2021/Stats in ML Project/data/X_train.txt'
y_train_fpath = 'C:/Users/Joy/Documents/Joy/Intercession 2021/Stats in ML Project/data/Y_train.txt'
x_test_fpath  = 'C:/Users/Joy/Documents/Joy/Intercession 2021/Stats in ML Project/data/X_test.txt'

# The first rows of the documents are the names of the features, so the usable data starts from
# the second line. Split up each vector.

with open(x_train_fpath) as f:
    next(f)
    x_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(y_train_fpath) as f:
    next(f)
    y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(x_test_fpath) as f:
    next(f)
    x_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    
print('x_train :\n', x_train, x_train.shape,'\n')
print('y_train :\n', y_train, y_train.shape,'\n')
print('x_test :\n', x_test, x_test.shape)

# Create a normalizatoin function that calculates a subset of data, the mean, and 
# standard deviations. Because the testing data calculations need to be normalized
# according to the mean and standard deviations of training data. 

def _normalize(x, train = True, specified_column = None, x_mean = None, x_std = None):
    # If columns are not specified, calculate all the columns. Otherwise, get
    # specific columns. 
    if specified_column == None:
        specified_column = np.arange(x.shape[1])
    
    # If training data is selected, only calculate rhe mean and std of the 
    # training data. 
    if train:
        # The reshape(1, -1) transposed the np.mean output into a row vector. 
        x_mean = np.mean(x[:, specified_column], axis = 0).reshape(1, -1)
        
        # calculate the std of specified columns. 
        x_std = np.std(x[:, specified_column], axis = 0).reshape(1, -1)
    x[:, specified_column] = (x[:, specified_column] - x_mean) / (x_std + 1e-8)
    
    return x, x_mean, x_std


# Splits x_train and y_train into training and validadtion sets based on the 
# specified ratio. Then, return the sizes of the arrays. 
def _train_split(x, y, validation_ratio = 0.25):
    
    train_size = int(len(x) * (1 - validation_ratio))
    
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]

# Normalize the training and testing data. Then print out the sizes.
x_train, x_mean, x_std = _normalize(x_train, train = True)
x_test, _, _ = _normalize(x_test, train = False, x_mean = x_mean, x_std = x_std)
x_training_set, y_training_set, x_validation_set, y_validation_set = _train_split(x_train, y_train, validation_ratio = 0.1)

print('x_training_set :', x_training_set.shape, '\n', x_training_set)
print('---------------------------------------------------------------')
print('y_training_set :', y_training_set.shape, '\n', y_training_set)
print('---------------------------------------------------------------')
print('x_validation_set :', x_validation_set.shape, '\n', x_validation_set)
print('---------------------------------------------------------------')
print('y_validation_set :', y_validation_set.shape, '\n', y_validation_set)
print('---------------------------------------------------------------')

# Randomly select the contents of the training and validation sets.
def _shuffle(x, y):
    randomize = np.arrange(len(x))
    np.ramdom.shuffle(randomize)
    
    return x[randomize], y[randomize]

# Apply the sigmoid function so that the output is always between 0 and 1. This
# calculates the probablity of a specific occurance. Use np.clip(a, a_min, a_max)
# to keep the outputs in a range so that they don't go on to -Inf or infinitely
# close to 1. 
def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

# The proposed function learned from the current iteration. x is the input, 
# w is the weight, b is the bias. This function goes through the sigmoid function 
# and generates a probability. 
# np.dot(x, w) takes the dot product of each dimension of x and each dimension 
# of w, which outputs a scalar. The bias is also updated through each iteration. 
def _f(x, w, b):
    return _sigmoid(np.dot(x, w) + b)

# Since the y_test outputs are either 0 or 1, and we determine it by seeing if
# the calculated probability exceeds 0.5, use rounding to get the prediction result
# of a specific function with w and b. Then, convert the results into integers.
def _predict(x, w, b):
    return np.round(_f(x, w, b)).astype(np.int)

# The predictions and labels are both binary with values of 0 or 1. 
# (y_predict - y_label) == 0 when the model predicts correctly, and ... == 1 otherwise. 
# Thus, np.mean(np.abs(y_predict - y_label)) divides the number of wrong answers by 
# the total number of instances to get the error rate. 
# Then, accuracy = (1 - error rate).
def _accuracy(y_predict, y_label):
    acc = 1 - np.mean(np.abs(y_predict - y_label))
    return acc

# Cross entropy aims to determine the goodness of a function. The smaller the loss
# value, the better. When plotting the loss vs. time graph, we also hope to see
# a gradual decrease in the loss values as functions update.
# Derivation of this formula (notes from in-class video lectures)
# Loss(w, b) = f(x1) * f(x2) * f(1-f(3)) * f(1-f(4)) * ...
#                ^ probability of outputting 1  ^Probability of outputting 0
# The best parameters, w and b, are denoted w* and b*. 
# w*, b* = argmax(Loss(w, b), which is equivalent to:
# w*, b* = argmin(-ln(Loss(w, b)))
# Loss(w, b) = -Σ[-y_label * ln(f(xn)) + (1 - y_label) * ln(1 - f(xn))]
# np.log() is equivalent to ln(). np.dot() implicitly calculates the summation 
# over n instances of data. 
# Therefore, the calculation is as follows:
def _cross_entropy_loss(y_predict, y_label):
    cross_entropy = -(np.dot(y_label, np.log(y_predict)) + np.dot((1 - y_label), np.log(1 - y_predict)))
    return cross_entropy

# From the loss function L(w,b) = -Σ(y_label*ln(y)+(1-y_label)*ln(1-y)), we 
# take the partial derivatives of w and b, respectively. 
# Gradient of w = -Σ(y_label - y) * x
# Gradient of b = -Σ(y_label - y)
def _gradient(x, y_label, w, b):
    y_predict = _f(x, w, b)
    w_gradient = -np.sum((y_label - y_predict) * x.T, 1)
    b_gradient = -np.sum(y_label - y_predict)
    
    return w_gradient, b_gradient

# Set variables for the sizes of different sets.
train_size = x_training_set.shape[0]
validation_size = x_validation_set.shape[0]
dim = x_training_set.shape[1]

# Initialize w and b.
w = np.zeros(dim)
b = np.zeros(1)

# Set the number of iterations and the learning rate.
max_iter = 600
learning_rate = 1

# Create empty arrays to store each new value. 
training_set_loss = []
training_set_acc = []
validation_set_loss = []
validation_set_acc = []

# Adagrad is a method that set custom learning rates that decrease as the number
# of iterations increase. 
w_adagrad = 1e-8
b_adagrad = 1e-8

# Apply the previously-defined functions to train 1000 times. 
for epoch in range(max_iter):
    w_gradient, b_gradient = _gradient(x_training_set, y_training_set, w, b)
    
    w_adagrad = w_adagrad + np.power(w_gradient, 2)
    b_adagrad = b_adagrad + np.power(b_gradient, 2)
    
    w = w - learning_rate * w_gradient / np.sqrt(w_adagrad)
    b = b - learning_rate * b_gradient / np.sqrt(b_adagrad)
    
    y_training_predict = _predict(x_training_set, w, b)
    y_probability = _f(x_training_set, w, b)
    acc = _accuracy(y_training_predict, y_training_set)
    loss = _cross_entropy_loss(y_probability, y_training_set)
    training_set_acc.append(acc)
    training_set_loss.append(loss)
    print('training_set_scc_%d  : %f \t training_set_loss_%d  : %f'%(epoch, acc, epoch, loss))
    
    y_validation_predict = _predict(x_validation_set, w, b)
    y_probability = _f(x_validation_set, w, b)
    acc = _accuracy(y_validation_predict, y_validation_set)
    loss = _cross_entropy_loss(y_probability, y_validation_set)
    validation_set_acc.append(acc)
    validation_set_loss.append(loss)

# This prints the final result of the training. With 600 iterations.
print('validation_set_scc_%d  : %f \t training_set_loss_%d  : %f'%(epoch, acc, epoch, loss))
print()

import matplotlib.pyplot as plt

# Plot the loss vs. iteration curve
plt.plot(training_set_loss)
plt.plot(validation_set_loss)
plt.title('Loss')
plt.legend(['training_set', 'validation_set'])
plt.savefig('loss_gd.png')
plt.show()

# Plot the accuracy vs. iteration curve
plt.plot(training_set_acc)
plt.plot(validation_set_acc)
plt.title('Accuracy')
plt.legend(['training_set', 'validation_set'])
plt.savefig('acc_gd.png')
plt.show()

# Now a model is trained. Write the .csv file of predictions. 
import csv
y_test_predict = _predict(x_test, w, b)
print(y_test_predict, y_test_predict.shape)

with open('income_prediction_outputs.csv', mode = 'w', newline = '') as f:
    csv_writer = csv.writer(f)
    header = ['id', 'label']
    print(header)
    csv_writer.writerow(header)
    for i in range(y_test_predict.shape[0]):
        row = [str(i), y_test_predict[i]]
        csv_writer.writerow(row)
        print(row)

        