import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import time
from sklearn.tree import DecisionTreeClassifier

# data is from https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models/notebook

# Load the Dataset
data = pd.read_csv("creditcard.csv") 

# Features (all columns except the 31st)
X = data.drop(data.columns[30], axis=1)
# Target column (31st column)
y = data.iloc[:, 30]  

# Create a transformer for numeric features
preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

# Extract values from the confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

# Logistic Regression
logistic_classifier = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', LogisticRegression())]) 

start_time = time.time()
logistic_classifier.fit(X_train, y_train)
lr_training_time = time.time() - start_time

start_time = time.time()
lr_predictions = logistic_classifier.predict(X_test)
lr_testing_time = time.time() - start_time

lr_cm = confusion_matrix(y_test, lr_predictions)

# Extract values from the confusion matrix
lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()

print("---------------------Logistic Regression---------------------")
print("Training Time:", lr_training_time)
print("Testing Time:", lr_testing_time)
print("Confusion Matrix:")
print(f"FP: {lr_fp}, TP: {lr_tp}")
print(f"TN: {lr_tn}, FN: {lr_fn}")

# Decision Tree
decision_tree_classifier = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', DecisionTreeClassifier(class_weight='balanced'))])

start_time = time.time()
decision_tree_classifier.fit(X_train, y_train)
dt_training_time = time.time() - start_time

start_time = time.time()
dt_predictions = decision_tree_classifier.predict(X_test)
dt_testing_time = time.time() - start_time

dt_cm = confusion_matrix(y_test, dt_predictions)

# Extract values from the confusion matrix
dt_tn, dt_fp, dt_fn, dt_tp = dt_cm.ravel()
print("---------------------Decision Tree---------------------")
print("Training Time:", dt_training_time)
print("Testing Time:", dt_testing_time)
print("Confusion Matrix:")
print(f"FP: {dt_fp}, TP: {dt_tp}")
print(f"TN: {dt_tn}, FN: {dt_fn}")