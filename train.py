import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(filepath):
    """Load preprocessed training data."""
    df = pd.read_csv(filepath)
    return df

def train_model(df):
    """Train a RandomForest model using the data."""
    # Separate features (X) and target (y)
    X = df.drop(columns=['target'], errors='ignore')
    
    # If dataset doesn't have a 'target' column, use last column as target
    if 'target' not in df.columns:
        y = df.iloc[:, -1]
    else:
        y = df['target']
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Model trained. Validation Accuracy: {acc:.4f}")
    
    return model

def save_model(model, out_dir):
    """Save the trained model to a file."""
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

def main():
    # Path to your train.csv file
    train_path = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Data\processed\train.csv"
    
    # Directory to save trained model
    model_dir = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Models"
    
    print("Loading training data...")
    df = load_data(train_path)
    print(f"Data shape: {df.shape}")
    
    print("Training model...")
    model = train_model(df)
    
    save_model(model, model_dir)

if __name__ == "__main__":
    main()
