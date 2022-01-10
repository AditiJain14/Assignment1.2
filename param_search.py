#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import scipy
import math


# In[51]:


import sklearn
from sklearn.metrics import f1_score


# In[52]:


from scipy.special import softmax


# In[53]:


train_path='/Users/aditijain/Documents/GitHub/COL341/Assignment_1/Assignment_1.2/data/train.csv'
test_path='/Users/aditijain/Documents/GitHub/COL341/Assignment_1/Assignment_1.2/data/test.csv'
train = pd.read_csv(train_path, index_col = 0)    
test = pd.read_csv(test_path, index_col = 0)
    
y_train = np.array(train['Length of Stay'])

train = train.drop(columns = ['Length of Stay'])

#Ensuring consistency of One-Hot Encoding

data = pd.concat([train, test], ignore_index = True)
cols = train.columns
cols = cols[:-1]
data = pd.get_dummies(data, columns=cols, drop_first=True)
data = data.to_numpy()
data=np.c_[np.ones(data.shape[0]),data]
X_train = data[:train.shape[0], :]
X_test = data[train.shape[0]:, :]


# In[54]:


y_encod=pd.get_dummies(y_train,columns=y_train)
y_encod.to_numpy()


# In[55]:


def const_grad_descent(iters,step,X,Y):
    W=np.zeros((X.shape[1],8))
    n=X.shape[0]
    for i in range(0,iters):
        print(i)
        Y_p=softmax(np.dot(X,W),axis=1)
        W_new=np.subtract(W,step/n*np.dot(np.transpose(X),np.subtract(Y_p,Y)))
        W=W_new
    return W


# In[56]:


def adaptive_grad_descent(iters,step,X,Y):
    W=np.zeros((X.shape[1],8))
    n=X.shape[0]
    for i in range(0,iters):
        Y_p=softmax(np.dot(X,W),axis=1)
        W_new=np.subtract(W,(step/(n*math.sqrt(i+1)))*np.dot(np.transpose(X),np.subtract(Y_p,Y)))
        W=W_new
    return W


# In[57]:


import matplotlib.pyplot as plt


# In[58]:


def graph(train,y,steps,Y_in):
    cons=[]
    adapt=[]
    for i in steps:
        print(i)
        W1=const_grad_descent(i,0.01,train,y)
        print("w done")
        y_Pred=softmax(np.dot(train,W1),axis=1)
        indexes=np.argmax(y_Pred,axis=1)+1
        index=pd.DataFrame(indexes)
        f=f1_score(Y_in,index,labels=[1,2,3,4,5,6,7,8],average="macro")
        print(f,"const")
        cons+=[f]
        W2=adaptive_grad_descent(i,0.01,train,y)
        y_Pred=softmax(np.dot(train,W2),axis=1)
        indexes=np.argmax(y_Pred,axis=1)+1
        index=pd.DataFrame(indexes)
        f=f1_score(Y_in,index,labels=[1,2,3,4,5,6,7,8],average="macro")
        adapt+=[f]
        print(f,"adapt")
    plt.plot(steps,cons)
    plt.plot(steps,adapt)
    plt.show()
    return 


# In[59]:


#steps=[0.01,0.02,0.03,0.05,0.1]
steps=[50,100,200,300,400,500]
steps1=[0.08,0.1]


# In[ ]:


graph(X_train,y_encod,steps,y_train)


# In[49]:


graph(X_train,y_encod,steps1,y_train)


# In[ ]:




