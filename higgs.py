
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


# Load the data
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('test.csv')

# Separate features, labels, and weights in training data
X_train = train_data.drop(['EventId', 'Label', 'Weight'], axis=1)
y_train = train_data['Label']
weights_train = train_data['Weight']

# Handle missing values if any
X_train.replace(-999.0, np.nan, inplace=True)
X_train.fillna(X_train.median(), inplace=True)

# Normalize weights
normalized_weights = weights_train / np.sum(weights_train)

# Split the data into training and testing sets
X_train_resample, X_test, y_train_resample, y_test, weights_train_resample, weights_test = train_test_split(
    X_train, y_train, weights_train, test_size=0.2, random_state=1
)

# Use LabelEncoder for string labels
label_encoder = LabelEncoder()
y_train_resample = label_encoder.fit_transform(y_train_resample)
y_test = label_encoder.transform(y_test)

# Standardize the features of the resampled data
scaler = StandardScaler()
X_train_resample = scaler.fit_transform(X_train_resample)
X_test = scaler.transform(X_test)

# Define the hyperparameters as keyword arguments
model = XGBClassifier(n_estimators=200, subsample=0.9,colsample_bynode=0.2)

# Fit the model on the resampled training data with sample weights
model.fit(X_train_resample, y_train_resample, sample_weight=weights_train_resample)

# Make predictions on the test set
y_pred = model.predict(X_test)




# Define the repeated stratified k-fold cross-validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Perform cross-validation and compute accuracy
accuracy_scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv)


# Calculate mean accuracy
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy using Repeated Stratified K-Fold Cross-Validation:", mean_accuracy)






