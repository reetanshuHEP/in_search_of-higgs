import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRFClassifier
import higgs
from higgs import*


# Now, let's hypertune the model to find the optimal number of trees
# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define the number of trees to consider
n_trees = [0.5,0.6,0.7,0.8,0.9]

# evaluate the models and store results
results = []
for n_tree in n_trees:
    model = XGBRFClassifier(n_estimators=250, subsample=0.9, colsample_bynode=0.2)
    scores = cross_val_score(model, X_train_resample, y_train_resample, scoring='accuracy', cv=cv, n_jobs=-1)
    results.append(scores)
    print('>%d trees: %.3f (%.3f)' % (n_tree, np.mean(scores), np.std(scores)))

# plot model performance for comparison
plt.boxplot(results, labels=n_trees, showmeans=True)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')

plt.title('XGBoostRF Model Performance')
plt.show()