#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.special import softmax
import math


# In[2]:


import sklearn


# In[3]:


import seaborn as sns


# In[4]:


train_path='/Users/aditijain/Documents/GitHub/COL341/Assignment_1/Assignment_1.2/data/train_large.csv'
test_path='/Users/aditijain/Documents/GitHub/COL341/Assignment_1/Assignment_1.2/data/train.csv'
train = pd.read_csv(train_path, index_col = 0)    
test = pd.read_csv(test_path, index_col = 0)
    
y_train = np.array(train['Length of Stay'])
Y=pd.get_dummies(y_train)


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


train.head()


# In[8]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[9]:


sns.heatmap(train.corr())


# In[10]:


for col in train.columns:
    print(col,np.corrcoef(train[col], train["Length of Stay"])[0,1])


# In[ ]:


to drop becoz low corr : Health Service Area, Operating Certificate Number, Facility Name, Type of Admission
repeated: 
high corr: APR MDC Code-APR DRG Code, Facility Id-Operating Number, Certificate: Zip Code, Birth Weight-Age Group


# high corr with Length of Stay : Age Group, Patient Disposition, APR Severity of Illness Code, Total Costs

# In[8]:


X=train.to_numpy()


# In[9]:


length_s=X[:,train.columns.get_loc('Total Costs')].reshape(X.shape[0],1)


# In[10]:


del train["Length of Stay"]


# In[11]:


del train["Operating Certificate Number"]
del train["Facility Id"]
del train["Zip Code - 3 digits"]
del train["Gender"]
del train["Race"]
del train["Ethnicity"]
del train["CCS Diagnosis Code"]
del train["CCS Procedure Code"]
del train["APR DRG Code"]
del train["APR Severity of Illness Code"]
del train["APR MDC Code"]
del train["APR Risk of Mortality"]
del train["Birth Weight"]
del train["Payment Typology 3"]
del train["Emergency Department Indicator"]
#from ass1


# In[48]:


del train["Operating Certificate Number"]
del train["Facility Name"]
del train["Hospital County"]
del train["Health Service Area"]
del train["Zip Code - 3 digits"]
del train["Gender"]
del train["Race"]
del train["Facility Id"]
del train["Ethnicity"]
del train["Type of Admission"]
del train["CCS Diagnosis Code"]
del train["CCS Procedure Code"]
del train["APR DRG Code"]
del train["APR MDC Code"]
del train["APR Risk of Mortality"]
del train["Birth Weight"]
del train["Payment Typology 3"]
del train["Emergency Department Indicator"]
del train["APR Medical Surgical Description"]
del train["Payment Typology 1"]
del train["Payment Typology 2"]


# In[12]:


train.shape


# In[13]:


for col in train.columns:
    print(col, train[col].unique().size)


# In[150]:


del train["Facility Name"]
del train["CCS Procedure Code"]
del train["APR DRG Code"]


# In[14]:


del train["Total Costs"]


# In[15]:


one_hot_encoded_data = pd.get_dummies(train, columns =train.columns)


# In[16]:


one_hot_encoded_data.shape


# In[17]:


for col in one_hot_encoded_data.columns:
    print(col)


# In[18]:


one_hot_encoded_data.drop(columns=['Health Service Area_1',
 'Health Service Area_2',
 'Health Service Area_7',
 'Health Service Area_8',
 'Hospital County_1',
 'Hospital County_2',
 'Hospital County_4',
 'Hospital County_5',
 'Hospital County_6',
 'Hospital County_7',
 'Hospital County_9',
 'Hospital County_10',
 'Hospital County_11',
 'Hospital County_12',
 'Hospital County_14',
 'Hospital County_15',
 'Hospital County_16',
 'Hospital County_17',
 'Hospital County_18',
 'Hospital County_20',
 'Hospital County_21',
 'Hospital County_22',
 'Hospital County_23',
 'Hospital County_24',
 'Hospital County_25',
 'Hospital County_27',
 'Hospital County_28',
 'Hospital County_31',
 'Hospital County_32',
 'Hospital County_34',
 'Hospital County_37',
 'Hospital County_38',
 'Hospital County_39',
 'Hospital County_40',
 'Hospital County_42',
 'Hospital County_44',
 'Hospital County_45',
 'Hospital County_46',
 'Hospital County_47',
 'Hospital County_48',
 'Hospital County_49',
 'Hospital County_50',
 'Hospital County_51',
 'Hospital County_52',
 'Hospital County_53',
 'Hospital County_56',
 'Hospital County_57',
 'Facility Name_2',
 'Facility Name_3',
 'Facility Name_4',
 'Facility Name_6',
 'Facility Name_17',
 'Facility Name_20',
 'Facility Name_21',
 'Facility Name_23',
 'Facility Name_24',
 'Facility Name_25',
 'Facility Name_29',
 'Facility Name_30',
 'Facility Name_31',
 'Facility Name_32',
 'Facility Name_33',
 'Facility Name_34',
 'Facility Name_36',
 'Facility Name_37',
 'Facility Name_38',
 'Facility Name_39',
 'Facility Name_42',
 'Facility Name_44',
 'Facility Name_46',
 'Facility Name_47',
 'Facility Name_50',
 'Facility Name_51',
 'Facility Name_53',
 'Facility Name_55',
 'Facility Name_56',
 'Facility Name_65',
 'Facility Name_68',
 'Facility Name_69',
 'Facility Name_74',
 'Facility Name_75',
 'Facility Name_80',
 'Facility Name_81',
 'Facility Name_82',
 'Facility Name_83',
 'Facility Name_84',
 'Facility Name_88',
 'Facility Name_92',
 'Facility Name_93',
 'Facility Name_98',
 'Facility Name_101',
 'Facility Name_105',
 'Facility Name_110',
 'Facility Name_113',
 'Facility Name_118',
 'Facility Name_119',
 'Facility Name_124',
 'Facility Name_125',
 'Facility Name_128',
 'Facility Name_129',
 'Facility Name_131',
 'Facility Name_133',
 'Facility Name_137',
 'Facility Name_138',
 'Facility Name_144',
 'Facility Name_146',
 'Facility Name_155',
 'Facility Name_156',
 'Facility Name_157',
 'Facility Name_160',
 'Facility Name_163',
 'Facility Name_165',
 'Facility Name_173',
 'Facility Name_174',
 'Facility Name_178',
 'Facility Name_179',
 'Facility Name_180',
 'Facility Name_183',
 'Facility Name_187',
 'Facility Name_188',
 'Facility Name_192',
 'Facility Name_194',
 'Facility Name_196',
 'Facility Name_203',
 'Facility Name_204',
 'Facility Name_212',
 'Age Group_3',
 'Type of Admission_2',
 'Type of Admission_4',
 'Patient Disposition_1',
 'Patient Disposition_4',
 'Patient Disposition_6',
 'Patient Disposition_7',
 'Patient Disposition_9',
 'Patient Disposition_10',
 'Patient Disposition_12',
 'Patient Disposition_15',
 'APR Medical Surgical Description_1',
 'Payment Typology 1_2',
 'Payment Typology 1_3',
 'Payment Typology 1_4',
 'Payment Typology 2_2',
 'Payment Typology 2_7',
 'Payment Typology 2_8'], axis=1, inplace=True)


# In[19]:


one_hot_encoded_data.to_numpy()


# In[ ]:





# In[20]:


X_extnd=np.c_[length_s,one_hot_encoded_data]


# In[21]:


X_extnd.shape


# In[22]:


def mini_batch_grad_descent(X,Y,mode,batches,epochs,step,alpha,beta):
    num=X.shape[0]//batches
    W=np.zeros((X.shape[1],8))
    n=X.shape[0]
    if(mode==1):
        for j in range(0,epochs):
            print(j)
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                Y_p=softmax(np.dot(mini_train,W),axis=1)
                W=np.subtract(W,(step/batches)*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))
    elif(mode==2):
        for j in range(0,epochs):
            print(j)
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                Y_p=softmax(np.dot(mini_train,W),axis=1)
                W=np.subtract(W,(step/(batches*math.sqrt(j+1)))*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))

    else:
        y_p=softmax(np.dot(X,W),axis=1)
        old=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
        delta=np.dot(np.transpose(X),np.subtract(y_p,Y))/n
        y_p=softmax(np.dot(X,W-step*delta),axis=1)
        new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
        for j in range(0,epochs):
            print(j)
            while(new>old-step*alpha*(np.linalg.norm(delta))**2):
                step=step*beta
                y_p=softmax(np.dot(X,W-step*delta),axis=1)
                new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                Y_p=softmax(np.dot(mini_train,W),axis=1)
                W=np.subtract(W,(step/batches)*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))
            y_p=softmax(np.dot(X,W),axis=1)
            old=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
            delta=np.dot(np.transpose(X),np.subtract(y_p,Y))/n
            y_p=softmax(np.dot(X,W-step*delta),axis=1)
            new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
        
    return W
            
        


# In[23]:


W=mini_batch_grad_descent(X_extnd,Y,2,500,100,0.1,0,0)


# In[ ]:


Y.shape


# In[ ]:


index.shape


# In[ ]:


y_pred_train=softmax(np.dot(X_extnd,W),axis=1)
indexes=np.argmax(y_pred_train,axis=1)+1
index=pd.DataFrame(indexes)
index


# In[61]:


y_Pred=softmax(np.dot(test,W),axis=1)
indexes=np.argmax(y_Pred,axis=1)+1
index=pd.DataFrame(indexes)
index


# In[131]:


from sklearn.metrics import f1_score


# In[136]:


f=f1_score(y_train,index,labels=[1,2,3,4,5,6,7,8],average="macro")
f


# In[133]:


f1_score


# In[134]:


index.shape


# In[ ]:




