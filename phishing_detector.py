import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("emails.csv")

# Features and labels
X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:")
print(cm)

# Test Email
email = ["Verify your bank account immediately"]
email_vector = vectorizer.transform(email)

prediction = model.predict(email_vector)

print("\nEmail:", email[0])
print("Prediction:", prediction[0])

# Save Report
with open("result.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))