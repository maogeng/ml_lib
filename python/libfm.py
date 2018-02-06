
# coding: utf-8

# In[85]:


from sklearn import datasets 
from numpy import random
from sklearn.model_selection import train_test_split

import numpy as np

TASK_REGRESSION = 0
TASK_CLASSIFICATION = 1

num_factor = 8

## 更新模型参数
def update(x, mult, s, v, w, w0, alpha):
    w0 -= alpha * mult
    for i in range(len(x)):
        w[i] -= alpha * (mult * x[i])
        
    for f in range(num_factor):
        for i in range(len(x)):
            grad = s[f] * x[i] - v[i,f] * x[i] * x[i]
            v[i,f] -= alpha * (mult * grad )

## 计算模型预测值
def predict(x, v, w, w0, s, sum_sqr):
    result = np.dot(w, x) + w0 # w*x
        
    # v*x 的 变形
    for f in range(num_factor):
        s[f] = 0
        sum_sqr[f] = 0
        for i in range(len(x)):
            d = v[i,f] * x[i]
            s[f] += d
            sum_sqr[f] += d*d
        result += 0.5 * (s[f] * s[f] - sum_sqr[f])
        
    return result

## 模型训练
def fit(X, y, s, sum_sqr, iter_num = 10, alpha = 0.01):
    num_samples = len(y)
    random.seed(1)
    w0 = random.random(1)
    v = random.random((X.shape[1], num_factor))
    w = random.random(X.shape[1])
    
    for i in range(iter_num):
        for j in range(len(y)):
            x = X[j]
            target = y[j]
            p = predict(x, v, w, w0, s, sum_sqr)
            
            mult = -target * (1.0 - 1.0/(1.0 + np.exp(-target * p)))
            
            update(x, mult, s, v, w, w0, alpha)
        
    return v, w, w0

## sigmoid 函数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

if __name__=='__main__':
    iris = datasets.load_iris()
    X = iris.data[:100]
    y = iris.target[:100]
    
    for i in range(len(y)):
        if y[i] <= 0.0:
            y[i]= -1.0
        else:
            y[i] = 1.0
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0)
    
    s = np.zeros(num_factor)
    sum_sqr = np.zeros(num_factor)
    
    v, w, w0 = fit(train_x, train_y, s, sum_sqr)
    
    for i in range(len(test_y)):
        label = sigmoid(predict(test_x[i], v, w, w0, s, sum_sqr))
        if label > 0.5:
            print(test_y[i], 1.0)
        else:
            print(test_y[i], -1.0)
    

