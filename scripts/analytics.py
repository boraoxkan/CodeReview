import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_FILE = "../dataset/raw/raw_data_batch_1.jsonl" 

def load_data(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
        
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                meta = entry.get("metadata", {})
                
                row = {
                    "language": meta.get("language", "Unknown"),
                    "bug_type": meta.get("bug_type", "Unknown"),
                    "difficulty": meta.get("difficulty", "Unknown"),
                    "input_length": len(entry.get("input", "")),
                }
                data.append(row)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

def analyze_dataset():
    print("--- STARTING DATASET ANALYSIS ---")
    df = load_data(DATA_FILE)
    
    if df is None or df.empty:
        print("No data available to analyze.")
        return

    # Stats
    print(f"\nTotal Samples: {len(df)}")
    print(f"Average Code Length: {df['input_length'].mean():.2f} chars")
    
    # Visualization
    plt.figure(figsize=(16, 10))

    # 1. Languages
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, y='language', order=df['language'].value_counts().index)
    plt.title('Distribution by Programming Language')

    # 2. Bug Types (Top 10)
    plt.subplot(2, 2, 2)
    bug_counts = df['bug_type'].value_counts().nlargest(10).index
    sns.countplot(data=df, y='bug_type', order=bug_counts)
    plt.title('Top 10 Bug Types')

    # 3. Difficulty
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='difficulty', order=['Beginner', 'Intermediate', 'Advanced'])
    plt.title('Difficulty Level')

    # 4. Length Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(df['input_length'], bins=30)
    plt.title('Code Length Distribution')

    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    analyze_dataset()