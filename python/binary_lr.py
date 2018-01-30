
# coding: utf-8

# In[59]:


from sklearn import datasets
from numpy import random

def sigmoid(x, weight):
    z = np.sum(x * weight)
    return 1.0/(1.0 + np.exp(-z))

def update(weight, x, y, alpha):
    predict = sigmoid(x, weight)
    gradient = (predict - y) * x
    weight = weight - alpha * gradient
    return weight

def classify(x, weight):
    z = sigmoid(x, weight)
    if z > 0.5:
        return 1
    else:
        return 0

def fit(X, y, iter_num, alpha):
    num_samples = len(y)
    
    random.seed(1)
    weight = random.random(X.shape[1])
    
    for i in range(iter_num):
        index = random.randint(num_samples)
        x = X[index]
        
        weight = update(weight, x, y[index], alpha)
    return weight
    
if __name__=='__main__':
    iter_num = 500 # iteration nums
    alpha = 0.01 # learning rate

    iris = datasets.load_iris()
    X = iris.data[:100]
    y = iris.target[:100]
    
    weight = fit(X, y, iter_num, alpha)
    
    for i in range(num_samples):
        x = X[i]
        print(sigmoid(x,weight), classify(x, weight), y[i])
        


