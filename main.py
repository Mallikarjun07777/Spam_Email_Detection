# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
# Download dataset from:
# https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(url, sep='\t', names=["label", "message"])

print("Dataset Preview:")
print(data.head())

# Convert labels to numeric (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Step 3: Text Vectorization
# -----------------------------
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Step 4: Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# Step 5: Prediction
# -----------------------------
y_pred = model.predict(X_test_vec)

# -----------------------------
# Step 6: Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 7: Custom Prediction
# -----------------------------
def predict_spam(message):
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)[0]
    
    if prediction == 1:
        return "SPAM ❌"
    else:
        return "NOT SPAM ✅"

# Test with your messages
print("\nCustom Predictions:")
print(predict_spam("Congratulations! You won a free lottery ticket"))
print(predict_spam("Hey, are we meeting today?"))