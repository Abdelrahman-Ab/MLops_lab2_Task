from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

def train_model(df):
    X = df.drop(columns=['target'], errors='ignore')
    y = df['target'] if 'target' in df.columns else df.iloc[:, -1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    return model

def main():
    train_path = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Data\processed\train.csv"
    model_dir = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Models"
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(train_path)
    model = train_model(df)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    print("Logistic Regression model saved successfully.")

if __name__ == "__main__":
    main()
