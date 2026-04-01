from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
texts = ["Real news", "Fake news", "True story", "Fake story"]
labels = [1, 0, 1, 0]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model trained and saved!")