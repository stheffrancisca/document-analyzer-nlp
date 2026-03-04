from loader import load_document
from preprocessing import preprocess_text
from features import build_tfidf
from model import train_model, evaluate_model, predict

def main():
    path = "data/sample.txt"

    documents = [
        "Este é um contrato jurídico formal com cláusulas legais.",
        "Relatório técnico de análise de dados e métricas.",
        "Nota fiscal e documento financeiro com valores."
    ]

    labels = ["juridico", "tecnico", "financeiro"]

    processed_docs = [preprocess_text(doc) for doc in documents]

    X, vectorizer = build_tfidf(processed_docs)

    model = train_model(X, labels)

    accuracy = evaluate_model(model, X, labels)
    print(f"Acurácia do modelo (treino): {accuracy:.2f}")

    text = load_document(path)
    processed_text = preprocess_text(text)

    prediction = predict(model, vectorizer, processed_text)

    print("Documento classificado como:", prediction)

if __name__ == "__main__":
    main()
