import json
import os

# Paths
INPUT_FILE = "../dataset/raw/raw_data_batch_1.jsonl"
OUTPUT_FILE = "../dataset/clean/clean_train_data.jsonl"

def clean_data():
    print("--- STARTING PREPROCESSING ---")
    
    # Check if raw file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Raw file not found at {INPUT_FILE}")
        return

    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    unique_inputs = set()
    valid_count = 0
    dropped_count = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        with open(INPUT_FILE, "r", encoding="utf-8") as in_f:
            for line in in_f:
                try:
                    data = json.loads(line)
                    
                    # Rule 1: Check Keys
                    if not all(key in data for key in ["instruction", "input", "output"]):
                        dropped_count += 1
                        continue
                        
                    # Rule 2: Check Length (Too short is usually garbage)
                    if len(data["input"]) < 20: 
                        dropped_count += 1
                        continue
                        
                    # Rule 3: Duplicate Check
                    code_signature = data["input"][:100].strip()
                    if code_signature in unique_inputs:
                        dropped_count += 1
                        continue
                    
                    unique_inputs.add(code_signature)
                    
                    # Write Clean Data
                    out_f.write(json.dumps(data) + "\n")
                    valid_count += 1
                    
                except json.JSONDecodeError:
                    dropped_count += 1
                    continue

    print(f"--- CLEANING COMPLETED ---")
    print(f"✅ Total Valid Samples: {valid_count}")
    print(f"🗑️ Dropped Samples: {dropped_count}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_data()