import pandas as pd
import glob
files = glob.glob("dataset.csv/*.csv")
print("CSV files found:")
for file in files:
    print(file)
df = pd.read_csv("dataset.csv/UNSW_NB15_training-set.csv")
print("Training dataset loaded!")
print(df.shape)
print(df.head())


X = df.drop(columns=["attack_cat", "label"])
y = df["label"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)

categorical_cols = X.select_dtypes(include=["str", "object"]).columns
print("Categorical columns:")
print(categorical_cols.tolist())

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print("After encoding:")
print("Features shape:", X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
#model1
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
print("Random Forest training completed!")

from sklearn.metrics import accuracy_score
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)
print("Random Forest Accuracy (%):", accuracy * 100)
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#model2
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)
lr.fit(X_train, y_train)
print("Logistic Regression training completed!")

y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Accuracy (%):", accuracy_lr * 100)

#model3
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
print("Naive Bayes training completed!")

y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", accuracy_nb)
print("Naive Bayes Accuracy (%):", accuracy_nb * 100)

#best model 
from sklearn.metrics import classification_report
print("\nRandom Forest:")
print(classification_report(y_test, y_pred))
print("\nLogistic Regression:")
print(classification_report(y_test, y_pred_lr))
print("\nNaive Bayes:")
print(classification_report(y_test, y_pred_nb))

import joblib
joblib.dump(rf, "random_forest_model.pkl")
print("Random Forest model saved successfully!")

sample = X_test.iloc[0:1]
prediction = rf.predict(sample)
print("Sample Prediction:", prediction[0])