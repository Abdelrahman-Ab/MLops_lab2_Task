import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()
    # Drop rows with missing values
    df = df.dropna()
    return df

def encode_categoricals(df):
    # Encode all categorical columns
    obj_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in obj_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def split_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def save_data(train_df, test_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

def main():
    # Update this path to your dataset location
    data_filepath = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Data\retail_sales_dataset.csv"
    
    # You can also set where you want to save train/test
    output_dir = r"C:\Users\peedo.abourayya\Desktop\Lab2TaskMlops\MLops_lab2_Task\Data\processed"

    print("📂 Loading dataset...")
    df = load_data(data_filepath)
    print(f"Initial shape: {df.shape}")

    df = clean_data(df)
    print(f"After cleaning: {df.shape}")

    df, _ = encode_categoricals(df)
    print("✅ Encoded categorical columns.")

    train_df, test_df = split_data(df)
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    save_data(train_df, test_df, output_dir)
    print(f"🎉 Saved train/test data in '{output_dir}'")

if __name__ == "__main__":
    main()
