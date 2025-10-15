import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def load_model(model_path):
    """Load the trained model."""
    model = joblib.load(model_path)
    return model

def load_test_data(test_path):
    """Load the test dataset."""
    df = pd.read_csv(test_path)
    return df

def evaluate_model(model, df):
    """Generate predictions and compute metrics."""
    X_test = df.drop(columns=['target'], errors='ignore')
    if 'target' not in df.columns:
        y_test = df.iloc[:, -1]
    else:
        y_test = df['target']

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_test, y_pred

def save_metrics(metrics, out_dir):
    """Save metrics as JSON file."""
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

def save_confusion_matrix(cm, labels, out_dir):
    """Save confusion matrix as an image."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

def main():
    # Paths
    model_path = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Models\model.pkl"
    test_path = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Data\processed\test.csv"
    output_dir = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Reports"

    print("Loading model...")
    model = load_model(model_path)

    print("Loading test data...")
    df_test = load_test_data(test_path)

    print("Evaluating model...")
    metrics, cm, y_test, y_pred = evaluate_model(model, df_test)

    save_metrics(metrics, output_dir)
    save_confusion_matrix(cm, sorted(set(y_test)), output_dir)

    print("Validation complete! Metrics and confusion matrix saved.")

if __name__ == "__main__":
    main()
