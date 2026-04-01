from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
texts = ["This is real news", "Fake news here", "This is true", "Totally fake"]
labels = [1, 0, 1, 0]  # 1 = Real, 0 = Fake

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test
test = ["This news is fake"]
test_vec = vectorizer.transform(test)
prediction = model.predict(test_vec)

print("Prediction:", "Real" if prediction[0] == 1 else "Fake")