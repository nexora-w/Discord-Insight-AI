import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/recruitment_sentences.csv", on_bad_lines='skip')

X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
report = classification_report(y_test, y_pred)
print(report)

# Save classification report to file
os.makedirs('reports', exist_ok=True)
with open('reports/classification_report.txt', 'w') as f:
    f.write(report)

# Plot and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png')
plt.close()

joblib.dump(model, "model/classifier.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("Model and vectorizer saved.")
