
# coding: utf-8

# In[14]:


from sklearn import datasets
from numpy import random
from sklearn.model_selection import train_test_split
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def update(weight, bias, x, y, alpha):
    predict = sigmoid(np.dot(x, weight))
    gradient = (predict - y) * x
    weight = weight - alpha * gradient
    bias = bias - alpha * (predict - y)
    return weight, bias

def classify(x, weight):
    z = sigmoid(np.dot(x, weight))
    if z > 0.5:
        return 1
    else:
        return 0

def fit(X, y, iter_num = 1000, alpha = 0.01):
    num_samples = len(y)
    
    random.seed(1)
    weight = random.random(X.shape[1])
    bias = 0.0
    
    for i in range(iter_num):
        index = random.randint(num_samples)
        x = X[index]
        
        weight, bias = update(weight, bias, x, y[index], alpha)
    return weight, bias
    
if __name__=='__main__':
    iris = datasets.load_iris()
    X = iris.data[:100]
    y = iris.target[:100]
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0)
    
#     clf = LogisticRegression(C=1.0, pen)
     
    weight, bias = fit(train_x, train_y)
    print(weight)
    print(bias)
    
    for i in range(len(test_y)):
        x = test_x[i]
        print(sigmoid(np.dot(x,weight)), classify(x, weight), test_y[i])
        


