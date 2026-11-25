from scam_ai import load_dataset, train_model

if __name__ == "__main__":
    metrics = train_model(load_dataset())
    print("Model saved to models/scam_model.joblib")
    for key in ["dataset_size", "train_size", "test_size", "accuracy", "precision", "recall", "f1"]:
        print(f"{key}: {metrics[key]}")
    print("Confusion matrix [legitimate, scam]:", metrics["confusion_matrix"])
