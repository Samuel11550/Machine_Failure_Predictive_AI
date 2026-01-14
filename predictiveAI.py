import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, 'ai4i2020.csv')
df = pd.read_csv(file_path)

df.columns = [
    'UDI', 'Product_ID', 'Type', 'AirTemp', 'ProcessTemp', 'RPM', 
    'Torque', 'ToolWear', 'Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]

#visualising the correlation of different parameters on the failure of the machine
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix['Failure'].sort_values(ascending=False))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title('Correlation Matrix Heatmap')
plt.show()

#--------DATA PREPOCESSING-------

#create new parameters to improve recall, these changes align more closely with real world physics

df['dT'] = df['ProcessTemp'] - df['AirTemp']
df['Power'] = df['Torque'] * df['RPM']


#turning the machine types into usable parameters using OneHotEncoder
ohe = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

ohe_transform = ohe.fit_transform(df[['Type']])
df = pd.concat([df, ohe_transform], axis=1).drop(columns=['Type'])

# Displaying the result after using the OneHotEncoder
print(df.head())

#dropping the last superflous columns
df = df.drop(columns=['UDI', 'Product_ID'])
print(df.head())

#training and testing data
x = df.drop(columns=['Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Failure']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#--------MODEL TRAINING------

#Random forest model
rf = RandomForestClassifier()

#tuning hyperparameters
param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4, 8],
    'bootstrap': [True, False]
}

tuner = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=50, 
    cv=3, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1,
    scoring='f1' 
)

tuner.fit(X_train, y_train)

print(f"Best F1 Score: {tuner.best_score_:.4f}")
print(f"Best Parameters: {tuner.best_params_}")

best_rf = tuner.best_estimator_

#Further model evaluation

y_pred = best_rf.predict(X_test)

best_rf.score(X_test, y_test)

print(classification_report(y_test, y_pred))

#visualising the confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#feature importance plot
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

names = [x.columns[i] for i in indices]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")

plt.bar(range(x.shape[1]), importances[indices])

plt.xticks(range(x.shape[1]), names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

#saving the model
joblib.dump(best_rf, 'predictiveRF.pkl')

