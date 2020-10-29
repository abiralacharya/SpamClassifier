# -*- coding: utf-8 -*-
"""
Case Study 1 for Machine Learning (EECS 5750 - Fall 2019)
Email Spam Classification

@author:Abiral Acharya, Amrit Niraula
"""

import numpy as np
import matplotlib.pyplot as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import KernelPCA

def aucCV(features,labels):
    
    model = GradientBoostingClassifier(learning_rate=0.1, 
                                       min_samples_split=50,
                                       n_estimators=100,
                                       min_samples_leaf=50,
                                       max_depth=9,
                                       max_features='sqrt',
                                       subsample=0.8)
    
    scores = cross_val_score(model, features, labels, cv=10,scoring='roc_auc')
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    
    model = GradientBoostingClassifier(learning_rate=0.1, 
                                       min_samples_split=50,
                                       n_estimators=100,
                                       min_samples_leaf=50,
                                       max_depth=9,
                                       max_features='sqrt',
                                       subsample=0.8)
    
    model.fit(trainFeatures, trainLabels)

    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:,1]
    
    return testOutputs
    
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data1 = np.loadtxt('spamTrain1.csv',delimiter=',')
    data2 = np.loadtxt('spamTrain2.csv',delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    data = np.r_[data1,data2]
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = data[:,:-1]
    labels = data[:,-1]
    
    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features,labels)))
    
    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    
    trainFeatures = features[0::2,:]
    trainLabels = labels[0::2]
    testFeatures = features[1::2,:]
    testLabels = labels[1::2]
    
       
    scaler = StandardScaler()
    trainFeatures = scaler.fit_transform(trainFeatures)
    testFeatures = scaler.transform(testFeatures)
    
    # Feature Extraction using Kernel PCA
    pca = KernelPCA(kernel="rbf",gamma=15,fit_inverse_transform=True)  
    trainFeatures_pca = pca.fit_transform(trainFeatures)  
    trainFeatures = pca.inverse_transform(trainFeatures_pca)
    
  
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    pl.subplot(2,1,1)
    pl.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    pl.xlabel('Sorted example number')
    pl.ylabel('Target')
    pl.subplot(2,1,2)
    pl.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    pl.xlabel('Sorted example number')
    pl.ylabel('Output (predicted target)')
    