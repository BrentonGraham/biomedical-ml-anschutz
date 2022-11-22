import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from NN_MultiLayer import NeuralNetwork_MultiLayer

# Set wkdir
WORK_DIR = os.getcwd()

# Import training, validation and test sets
trainingDataDF = pd.read_excel(f'{WORK_DIR}/GeneExpressionCancer_training.xlsx').to_numpy()
validationDataDF = pd.read_excel(f'{WORK_DIR}/GeneExpressionCancer_validation.xlsx').to_numpy()
testDataDF = pd.read_excel(f'{WORK_DIR}/GeneExpressionCancer_test.xlsx').to_numpy()

# Separate into labels/features
trainingLabels = trainingDataDF[:,-1]
trainingFeatures = trainingDataDF[:,:-1]
validationLabels = validationDataDF[:,-1]
validationFeatures = validationDataDF[:,:-1]
testLabels = testDataDF[:,-1]
testFeatures = testDataDF[:,:-1]

# Normalizing features; fit/transform on training set then transform validation
scaler = sklearn.preprocessing.StandardScaler()
trainingData_scaled = scaler.fit_transform(trainingFeatures)
validationData_scaled = scaler.transform(validationFeatures)
testData_scaled = scaler.transform(testFeatures)

# Instantiate NN
nn_multiLayer = NeuralNetwork_MultiLayer(nFeatures=len(trainingDataDF[:, :-1][0]))

# Train NN
nn_multiLayer = nn_multiLayer.trainModel(
    trainingFeatures=trainingFeatures,
    trainingLabels=trainingLabels,
    validationFeatures=validationFeatures,
    validationLabels=validationLabels)
