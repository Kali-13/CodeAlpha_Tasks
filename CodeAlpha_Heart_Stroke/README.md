

Heart Disease Prediction using Machine Learning
Table of Contents
Project Overview

Dataset

Methodology

1. Data Exploration and Cleaning

2. Data Preprocessing

3. Model Building

4. Hyperparameter Tuning

5. Model Evaluation

Results

Usage

Files in this Repository

Project Overview
This project aims to predict the presence of heart disease in patients based on a set of medical attributes. Various machine learning classification models were trained and evaluated to determine the most effective algorithm for this prediction task. The final model, a tuned Random Forest Classifier, demonstrates high accuracy in identifying patients with heart disease.

Dataset
The project utilizes the "Heart Failure Prediction" dataset. This dataset contains 918 observations and 12 features related to patient health.

Features:

Age: Age of the patient [years]

Sex: Sex of the patient [M: Male, F: Female]

ChestPainType: Type of chest pain [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]

RestingBP: Resting blood pressure [mm Hg]

Cholesterol: Serum cholesterol [mm/dl]

FastingBS: Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]

RestingECG: Resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality, LVH: showing probable or definite left ventricular hypertrophy]

MaxHR: Maximum heart rate achieved [Numeric value between 60 and 202]

ExerciseAngina: Exercise-induced angina [Y: Yes, N: No]

Oldpeak: ST depression induced by exercise relative to rest

ST_Slope: The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]

HeartDisease: The target variable, indicating the presence of heart disease [1: Heart Disease, 0: Normal]

Methodology
The project follows a systematic approach from data exploration to model deployment.

1. Data Exploration and Cleaning
Initial Analysis: The dataset was loaded and inspected for basic properties like data types, null values, and duplicates. No missing values or duplicate rows were found.

Handling Zero Values: The Cholesterol and RestingBP columns contained a significant number of zero values, which are medically implausible. These zeros were treated as missing data and were imputed with the mean of the non-zero values for each respective column. This ensures a more realistic data distribution for model training.

2. Data Preprocessing
Encoding Categorical Features:

Binary features (Sex, ExerciseAngina) were mapped to numerical values (0 and 1).

Multi-class categorical features (ChestPainType, RestingECG, ST_Slope) were converted into numerical format using one-hot encoding.

Feature Scaling:

Numerical features (Age, RestingBP, Cholesterol, MaxHR, Oldpeak) were standardized using StandardScaler from scikit-learn. This ensures that all features contribute equally to the model's performance by scaling them to have a mean of 0 and a standard deviation of 1.

3. Model Building
Four different classification models were initially trained on the preprocessed data to establish a baseline performance:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest Classifier

4. Hyperparameter Tuning
To optimize performance, GridSearchCV with 5-fold cross-validation was used to find the best combination of hyperparameters for each model.

Logistic Regression: Tuned C and penalty.

Best Parameters: {'C': 10, 'penalty': 'l2'}

K-Nearest Neighbors: Tuned n_neighbors, weights, and metric.

Best Parameters: {'metric': 'manhattan', 'n_neighbors': 13, 'weights': 'distance'}

Support Vector Machine: Tuned C, gamma, and kernel.

Best Parameters: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}

Random Forest Classifier: Tuned n_estimators, max_depth, min_samples_split, min_samples_leaf, and bootstrap.

Best Parameters: {'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}

5. Model Evaluation
The models were evaluated on the test set using Accuracy and F1-Score as the primary metrics. The Random Forest Classifier consistently provided the best performance both before and after hyperparameter tuning.

Results
After hyperparameter tuning, the Random Forest model was selected as the final model due to its superior performance.

Final Model: Random Forest Classifier

Accuracy: 87.5%

F1-Score: 89.0%

Classification Report:

               precision    recall  f1-score   support

           0       0.83      0.88      0.86        77
           1       0.91      0.87      0.89       107

    accuracy                           0.88       184
   macro avg       0.87      0.88      0.87       184
weighted avg       0.88      0.88      0.88       184
Usage
To run this project, you need to have Python and the following libraries installed:

pandas