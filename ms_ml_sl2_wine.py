# -*- coding: utf-8 -*-
"""MS_ML_SL2-Wine.ipynb"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('winequality-red.csv')

# Data Preprocessing
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
gscaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'kNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Neural Network': MLPClassifier(max_iter=1000)
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        'report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }

# Cross-Validation Scores
cv_results = {}
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_results[model_name] = cv_scores

# Learning Curves
# (add code for learning curves if required)

# Performance Comparisons
for model_name, result in results.items():
    print(f"{model_name} Results:")
    print(result['report'])
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# (add code for visualizing cross-validation scores if required)