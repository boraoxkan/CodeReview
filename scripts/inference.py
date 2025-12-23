from unsloth import FastLanguageModel
import torch
import json

# --- CONFIGURATION ---
MODEL_PATH = "../models/github_agent_v1" # Eğittiğimiz modelin yolu

# Test Code (SQL Injection Example)
TEST_CODE = """
user_input = request.args.get('username')
cursor = db.cursor()
# Kullanıcıdan gelen veri direkt sorguya eklenmiş!
query = "SELECT * FROM users WHERE username = '" + user_input + "'"
cursor.execute(query)
"""

def test_model():
    print(f"📦 Loading Model from {MODEL_PATH}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze this Python code for defects.

### Input:
{}

### Response:
"""
    
    inputs = tokenizer(
        [alpaca_prompt.format(TEST_CODE)], 
        return_tensors = "pt"
    ).to("cuda")

    print("🤖 Agent is analyzing...")
    outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
    
    response_text = tokenizer.batch_decode(outputs)[0]
    json_part = response_text.split("### Response:")[-1].replace("<|endoftext|>", "").strip()
    
    print("\n--- AGENT REPORT ---")
    print(json_part)

if __name__ == "__main__":
    test_model()