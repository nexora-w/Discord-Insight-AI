import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

test_df = pd.read_csv("test/test.csv")

model = joblib.load("model/classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

X_test = vectorizer.transform(test_df["sentence"])
y_true = test_df["label"]

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_true, y_pred)
print(f"Model accuracy on test set: {accuracy:.4f}")