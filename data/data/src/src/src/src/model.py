from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def train_model(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy

def predict(model, vectorizer, text):
    X_new = vectorizer.transform([text])
    return model.predict(X_new)[0]
