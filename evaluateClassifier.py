# -*- coding: utf-8 -*-
"""
Script used to evaluate classifier accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import roc_auc_score,roc_curve
from classifySpam import predictTest
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

nRuns = 10
desiredFPR = 0.01
trainDataFilename = 'spamTrain1.csv'
testDataFilename = 'spamTrain2.csv'
#testDataFilename = 'spamTest.csv'

def tprAtFPR(labels,outputs,desiredFPR):
    fpr,tpr,thres = roc_curve(labels,outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr<=desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex+1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex+1]
    tprAt = ((tprAbove-tprBelow)/(fprAbove-fprBelow)*(desiredFPR-fprBelow) 
             + tprBelow)
    return tprAt,fpr,tpr

pl.ion()

trainData = np.loadtxt(trainDataFilename,delimiter=',')
testData = np.loadtxt(testDataFilename,delimiter=',')

# Randomly shuffle rows of training and test sets then separate labels
# (last column)
shuffleIndex = np.arange(np.shape(trainData)[0])
np.random.shuffle(shuffleIndex)
trainData = trainData[shuffleIndex,:]
trainFeatures = trainData[:,:-1]
trainLabels = trainData[:,-1]

shuffleIndex = np.arange(np.shape(testData)[0])
np.random.shuffle(shuffleIndex)
testData = testData[shuffleIndex,:]
testFeatures = testData[:,:-1]
testLabels = testData[:,-1]

scaler = StandardScaler()
trainFeatures = scaler.fit_transform(trainFeatures)
testFeatures = scaler.transform(testFeatures)
    
# Feature Extraction using Kernel PCA
pca = KernelPCA(kernel="rbf",gamma=15,fit_inverse_transform=True)  
trainFeatures_pca = pca.fit_transform(trainFeatures)  
trainFeatures = pca.inverse_transform(trainFeatures_pca)
testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
aucTestRun = roc_auc_score(testLabels,testOutputs)
tprAtDesiredFPR,fpr,tpr = tprAtFPR(testLabels,testOutputs,desiredFPR)

pl.plot(fpr,tpr)

print(f'Test set AUC: {aucTestRun}')
print(f'Mean TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}')
pl.xlabel('False positive rate')
pl.ylabel('True positive rate')
pl.title('ROC curve for spam detector')    
pl.show()
