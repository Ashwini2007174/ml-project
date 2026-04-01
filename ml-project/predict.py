import pickle
# Load model
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)
# Input
text = input("Enter news: ")
# Predict
X = vectorizer.transform([text])
prediction = model.predict(X)

print("Prediction:", "Real" if prediction[0] == 1 else "Fake")