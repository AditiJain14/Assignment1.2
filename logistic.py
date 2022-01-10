#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.special import softmax
import math


# In[2]:


import sys
import datetime


# In[ ]:


def preprocess(train,test):
    y_train = np.array(train['Length of Stay'])
    train = train.drop(columns = ['Length of Stay'])


    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    data=np.c_[np.ones(data.shape[0]),data]
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    y_encod=pd.get_dummies(y_train,columns=y_train)
    y_encod=y_encod.to_numpy()
    return X_train,X_test,y_encod


# In[ ]:


def LG_A(train_Data,test_Data,params,outputfile,weightfile):
    train,test,y_train=preprocess(train_Data,test_Data)
    
    def const_grad_descent(iters,step,X,Y):
        W=np.zeros((X.shape[1],8))
        n=X.shape[0]
        for i in range(0,iters):
            Y_p=softmax(np.dot(X,W),axis=1)
            W_new=np.subtract(W,step/n*np.dot(np.transpose(X),np.subtract(Y_p,Y)))
            W=W_new
        return W

    def adaptive_grad_descent(iters,step,X,Y):
        W=np.zeros((X.shape[1],8))
        n=X.shape[0]
        for i in range(0,iters):
            Y_p=softmax(np.dot(X,W),axis=1)
            W_new=np.subtract(W,(step/(n*math.sqrt(i+1)))*np.dot(np.transpose(X),np.subtract(Y_p,Y)))
            W=W_new
        return W

    def backsearch_grad_descent(iters,step,X,Y,alpha,beta):
        W=np.zeros((X.shape[1],8))
        n=X.shape[0]
        y_p=softmax(np.dot(X,W),axis=1)
        old=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
        delta=np.dot(np.transpose(X),np.subtract(y_p,Y))/n
        y_p=softmax(np.dot(X,W-step*delta),axis=1)
        new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
        for i in range(0,iters):
            while(new>old-step*alpha*(np.linalg.norm(delta))**2):
                step=step*beta
                y_p=softmax(np.dot(X,W-step*delta),axis=1)
                new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
            old=new
            step=s0
        #delta=np.dot(np.transpose(X),np.subtract(y_p,Y))/n
            W=W-step*delta
            y_p=softmax(np.dot(X,W),axis=1)
            delta=np.dot(np.transpose(X),np.subtract(y_p,Y))/n
            y_p=softmax(np.dot(X,W-step*delta),axis=1)
            new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
        return W
    if(params[0]==1):
        W=const_grad_descent(params[1],params[2],train,y_train)
    elif(params[0]==2):
        W=adaptive_grad_descent(params[1],params[2],train,y_train)
    elif(params[0]==3):
        W=backsearch_grad_descent(params[1],params[2],train,y_train,)
    y_Pred=softmax(np.dot(X_test,W),axis=1)
    W=W.flatten()
    saved_out = list(zip(W))
    df = pd.DataFrame(saved_out)
    df.to_csv(weightfile,index = False, header=False)
    indexes=np.argmax(y_Pred,axis=1)+1
    index=pd.DataFrame(indexes)
    index.to_csv(outputfile,index = False, header=False)
    return outputfile,weightfile


# In[ ]:


def LG_B(train_Data,test_Data,params,output_file,output_weight):
    train,test,y_train=preprocess(train_Data,test_Data)
    def mini_batch_grad_descent(X,Y,mode,batches,epochs,step,alpha,beta):
        num=X.shape[0]//batches
        W=np.zeros((X.shape[1],8))
        n=X.shape[0]
        if(mode==1):
            for j in range(0,epochs):
                for i in range(0,num):
                    mini_train=X[i*batches:(i+1)*batches]
                    mini_out=Y[i*batches:(i+1)*batches]
                    Y_p=softmax(np.dot(mini_train,W),axis=1)
                    W=np.subtract(W,(step/batches)*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))
        elif(mode==2):
            for j in range(0,epochs):
                for i in range(0,num):
                    mini_train=X[i*batches:(i+1)*batches]
                    mini_out=Y[i*batches:(i+1)*batches]
                    Y_p=softmax(np.dot(mini_train,W),axis=1)
                    W=np.subtract(W,(step/(batches*math.sqrt(j+1)))*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))

        else:
            s0=step
            y_p=softmax(np.dot(X,W),axis=1)
            old=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
            delta=np.dot(np.transpose(X),np.subtract(y_p,Y))/n
            y_p=softmax(np.dot(X,W-step*delta),axis=1)
            new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
            for j in range(0,epochs):
                step=s0
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
    if(params[0]==1):
        W=mini_batch_grad_descent(train,y_train,params[0],params[2],params[1],params[3],0,0)
    elif(params[0]==2):
        W=mini_batch_grad_descent(train,y_train,params[0],params[2],params[1],params[3],0,0)
    elif(params[0]==3):
        W=mini_batch_grad_descent(train,y_train,params[0],params[2],params[1],params[3],params[4],params[5])
    y_Pred=softmax(np.dot(X_test,W),axis=1)
    W=W.flatten()
    saved_out = list(zip(W))
    df = pd.DataFrame(saved_out)
    df.to_csv(weightfile,index = False, header=False)
    indexes=np.argmax(y_Pred,axis=1)+1
    index=pd.DataFrame(indexes)
    index.to_csv(outputfile,index = False, header=False)
    return outputfile,weightfile
        


# In[ ]:


def LG_C(train_Data,test_Data,outputfile,weightfile):
    train,test,y_train=preprocess(train_Data,test_Data)
    def mini_batch_grad_descent(X,Y,mode,batches,epochs,step,alpha,beta,outputfile,weightfile):
        t0=datetime.datetime.now()
        num=X.shape[0]//batches
        W=np.zeros((X.shape[1],8))
        n=X.shape[0]
        if(mode==1):
            for j in range(0,epochs):
                for i in range(0,num):
                    mini_train=X[i*batches:(i+1)*batches]
                    mini_out=Y[i*batches:(i+1)*batches]
                    Y_p=softmax(np.dot(mini_train,W),axis=1)
                    W=np.subtract(W,(step/batches)*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))
        elif(mode==2):
            for j in range(0,epochs):
                for i in range(0,num):
                    mini_train=X[i*batches:(i+1)*batches]
                    mini_out=Y[i*batches:(i+1)*batches]
                    Y_p=softmax(np.dot(mini_train,W),axis=1)
                    W=np.subtract(W,(step/(batches*math.sqrt(j+1)))*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))

        else:
            s0=step
            y_p=softmax(np.dot(X,W),axis=1)
            old=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
            delta=np.dot(np.transpose(X),np.subtract(y_p,Y))/n
            y_p=softmax(np.dot(X,W-step*delta),axis=1)
            new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
            for j in range(0,epochs):
                step=s0
                while(new>old-step*alpha*(np.linalg.norm(delta))**2):
                    step=step*beta
                    y_p=softmax(np.dot(X,W-step*delta),axis=1)
                    new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
                t1=datetime.datetime.now()
                if(t1-t0>15):
                    y_Pred=softmax(np.dot(X_test,W),axis=1)
                    W=W.flatten()
                    saved_out = list(zip(W))
                    df = pd.DataFrame(saved_out)
                    df.to_csv(weightfile,index = False, header=False)
                    indexes=np.argmax(y_Pred,axis=1)+1
                    index=pd.DataFrame(indexes)
                    index.to_csv(outputfile,index = False, header=False)
        
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
    W=mini_batch_grad_descent(train,y_train,3,500,500,2.5,0.4,0.8,outputfile,weightfile)
    y_Pred=softmax(np.dot(X_test,W),axis=1)
    W=W.flatten()
    saved_out = list(zip(W))
    df = pd.DataFrame(saved_out)
    df.to_csv(weightfile,index = False, header=False)
    indexes=np.argmax(y_Pred,axis=1)+1
    index=pd.DataFrame(indexes)
    index.to_csv(outputfile,index = False, header=False)
    return outputfile,weightfile
        
    


# In[ ]:


def LG_D(train_Data,test_Data,outputfile,weightfile):
    #Y=train_Data["Length of Stay"]
    #del train_Data["Length of Stay"]

    def predrop(train_Data):
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
        return train_Data
    train_Data=predrop(train_Data)
    test_Data=predrop(test_Data)
    def drop(one_hot_encoded_data):
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
        return one_hot_encoded_data
    def feature(train_Data):
        X=train_Data.to_numpy()
        length_s=X[:,train_Data.columns.get_loc('Total Costs')].reshape(X.shape[0],1)
        colmn=[]
        for i in train_Data.columns:
            if i!="Total Costs":
                colmn+=[i]
        
        one_hot_encoded_data = pd.get_dummies(train_Data, columns =colmn)
        print(one_hot_encoded_data.columns)
        one_hot_encoded_data=drop(one_hot_encoded_data)
        #one_hot_encoded_data = one_hot_encoded_data[:-212]
        one_hot_encoded_data.to_numpy()
        X_t=np.c_[one_hot_encoded_data,length_s]
        return X_t
    size=test_Data.shape[0]
    train_Data=train_Data.append(test_Data)
    total=feature(train_Data)
    X_test=total[-size:]
    X_train=total[:-size]
    #X_train=feature(train_Data)
    #X_test=feature(test_Data)
    #X_test=X_test[:-212]
    def predict(X,Y,test_x,outputfile,weightfile,W,batches,epochs,step,alpha,beta):
        def mini_batch_grad_descent(X,Y,mode,batches,epochs,step,alpha,beta,X_test,outputfile,weightfile):
            t0=datetime.datetime.now()
            num=X.shape[0]//batches
            W=np.zeros((X.shape[1],8))
            n=X.shape[0]
            if(mode==1):
                for j in range(0,epochs):
                    for i in range(0,num):
                        mini_train=X[i*batches:(i+1)*batches]
                        mini_out=Y[i*batches:(i+1)*batches]
                        Y_p=softmax(np.dot(mini_train,W),axis=1)
                        W=np.subtract(W,(step/batches)*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))
            elif(mode==2):
                for j in range(0,epochs):
                    for i in range(0,num):
                        mini_train=X[i*batches:(i+1)*batches]
                        mini_out=Y[i*batches:(i+1)*batches]
                        Y_p=softmax(np.dot(mini_train,W),axis=1)
                        W=np.subtract(W,(step/(batches*math.sqrt(j+1)))*np.dot(np.transpose(mini_train),np.subtract(Y_p,mini_out)))

            else:
                s0=step
                y_p=softmax(np.dot(X,W),axis=1)
                old=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
                delta=np.dot(np.transpose(X),np.subtract(y_p,Y))/n
                y_p=softmax(np.dot(X,W-step*delta),axis=1)
                new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
                for j in range(0,epochs):
                    step=s0
                    while(new>old-step*alpha*(np.linalg.norm(delta))**2):
                        step=step*beta
                        y_p=softmax(np.dot(X,W-step*delta),axis=1)
                        new=(-1)*np.sum(np.sum(np.multiply(Y,np.log(y_p))))/n
                    t1=datetime.datetime.now()
                    if(t1-t0>45):
                        y_Pred=softmax(np.dot(X_test,W),axis=1)
                        W=W.flatten()
                        saved_out = list(zip(W))
                        df = pd.DataFrame(saved_out)
                        df.to_csv(weightfile,index = False, header=False)
                        indexes=np.argmax(y_Pred,axis=1)+1
                        index=pd.DataFrame(indexes)
                        index.to_csv(outputfile,index = False, header=False)
        
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
        W=mini_batch_grad_descent(X,Y,mode,batches,epochs,step,alpha,beta)
        y_Pred=softmax(np.dot(test_x,W),axis=1)
        W=W.flatten()
        saved_out = list(zip(W))
        df = pd.DataFrame(saved_out)
        df.to_csv(weightfile,index = False, header=False)
        indexes=np.argmax(y_Pred,axis=1)+1
        index=pd.DataFrame(indexes)
        index.to_csv(outputfile,index = False, header=False)
        return outputfile,weightfile
    
    return predict(X_train,Y,X_test,outputfile,weightfile,W,500,200,2.5,0.5,0.9)


# In[ ]:


if(len(sys.argv)==7 and sys.argv[1]=="a"):
#logistic.py a trainfile.csv testfile.csv param.txt outputfile.txt weightfile.txt
    train_Data=pd.read_csv(sys.argv[2],index_col=[0])
    test_Data=pd.read_csv(sys.argv[3],index_col=[0])
    with open(sys.argv[4]) as sample:
        lines = sample.readlines()
    params=[] #(mode,iters,step(or [step,alpha,beta]))
    params.append(int(lines[0].strip()))
    params.append(int(lines[2].strip()))
    m=lines[1].strip().split(sep=",")
    for i in m:
        params.append(float(i))
    output_file=sys.argv[4]
    output_weight=sys.argv[5]
    LG_A(train_Data,test_Data,params,output_file,output_weight)
elif (len(sys.argv)==7 and sys.argv[1]=="b"):
#logistic.py b trainfile.csv testfile.csv param.txt outputfile.txt weightfile.txt
    train_Data=pd.read_csv(sys.argv[2],index_col=[0])
    test_Data=pd.read_csv(sys.argv[3],index_col=[0])
    with open(sys.argv[4]) as sample:
        lines = sample.readlines()
    params=[] #(mode epochs batchsize step[step,alpha,beta])
    params.append(int(lines[0].strip()))
    params.append(int(lines[2].strip()))
    params.append(int(lines[3].strip()))
    m=lines[1].strip().split(sep=",")
    for i in m:
        params.append(float(i))
    output_file=sys.argv[5]
    output_weight=sys.argv[6]
    LG_B(train_Data,test_Data,params,output_file,output_weight)
elif(len(sys.argv)==6 and sys.argv[1]=="c"):
#logistic.py c trainfile.csv testfile.csv outputfile.txt weightfile.txt
    train_Data=pd.read_csv(sys.argv[2],index_col=[0])
    test_Data=pd.read_csv(sys.argv[3],index_col=[0])
    output_file=sys.argv[4]
    output_weight=sys.argv[5]
    LG_C(train_Data,test_Data,output_file,output_weight)
elif(len(sys.argv)==6 and sys.argv[1]=="d"):
#logistic.py d trainfile.csv testfile.csv outputfile.txt weightfile.txt
    train_Data=pd.read_csv(sys.argv[2],index_col=[0])
    test_Data=pd.read_csv(sys.argv[3],index_col=[0])
    output_file=sys.argv[4]
    output_weight=sys.argv[5]
    LG_D(train_Data,test_Data,output_file,output_weight)
else:
    print("invalid",len(sys.argv))

