import requests
import json
import random
import os
from tqdm import tqdm
import time

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
TEACHER_MODEL = "qwen2.5:32b" # Using the optimized model for your 64GB RAM
OUTPUT_FILE = "../dataset/raw/raw_data_batch_1.jsonl"
TOTAL_SAMPLES = 500  # Set your target here (e.g., 500 or 1000)

LANGUAGES = ["Python", "JavaScript", "Java", "C++", "Go", "SQL", "C#", "Rust", "TypeScript"]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]

# --- MASSIVE BUG LIST ---
BUG_TYPES = [
    # Security (OWASP)
    "SQL Injection vulnerability",
    "Cross-Site Scripting (XSS)",
    "Hardcoded sensitive credentials (API keys, passwords)",
    "Insecure Deserialization",
    "Path Traversal vulnerability",
    "Command Injection",
    "Weak Cryptographic Hash (MD5/SHA1)",
    "Missing Rate Limiting",
    
    # Performance
    "N+1 Query Problem (Database)",
    "Infinite Loop inside a logic block",
    "Memory Leak due to unclosed resources",
    "Inefficient String Concatenation inside loops",
    "Blocking I/O operation in the main thread",
    "Unoptimized Recursive Function (Stack Overflow risk)",
    
    # Logic & Quality
    "Off-by-one Error (IndexOutOfBounds)",
    "Division by Zero without check",
    "Shadowing variables in nested scope",
    "Swallowed Exception (Empty catch block)",
    "Race Condition in multi-threaded code",
    "Deadlock scenario",
    "Floating point precision error (money calculation)",
    "Magic Numbers instead of constants",
    "Spaghetti Code (High Cyclomatic Complexity)",
    "Improper Null Handling (NullPointerException risk)",
    "Use of Deprecated API"
]

def call_ollama(prompt, system_prompt, temp=0.7):
    payload = {
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "temperature": temp,
        "format": "json"
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json().get('response', '')
    except Exception as e:
        # print(f"API Error: {e}") 
        return None

def generate_sample():
    lang = random.choice(LANGUAGES)
    bug = random.choice(BUG_TYPES)
    diff = random.choice(DIFFICULTIES)
    
    # STEP 1: Generate Bad Code (High Temp 1.0 for Creativity)
    saboteur_prompt = f"""
    Write a {lang} code snippet containing a specific '{bug}'.
    Difficulty level: {diff}.
    Return ONLY a JSON object with a single key 'code'.
    Example: {{ "code": "print('hello')" }}
    """
    
    raw_response = call_ollama(saboteur_prompt, "You are a developer creating buggy code for testing.", temp=1.0)
    
    if not raw_response: return None
    
    try:
        bad_code = json.loads(raw_response).get("code", "")
    except:
        return None 

    if len(bad_code) < 30: return None 

    # STEP 2: Analyze Code (Low Temp 0.3 for Strict Formatting)
    analyst_prompt = f"""
    Analyze this {lang} code. It contains a {bug}.
    Code:
    {bad_code}

    Output a JSON response with exactly these keys:
    - code_quality_score (integer 0-100)
    - critical_issues (list of strings)
    - suggestions (list of strings)
    - fixed_code (string)
    """
    
    analysis_raw = call_ollama(analyst_prompt, "You are a Senior Code Reviewer.", temp=0.3)
    
    if not analysis_raw: return None
    
    try:
        analysis_json = json.loads(analysis_raw)
        
        # Unsloth Training Data Format
        return {
            "instruction": f"Analyze this {lang} code for defects.",
            "input": bad_code,
            "output": json.dumps(analysis_json), # Dump to string for training
            "metadata": {
                "language": lang, 
                "difficulty": diff, 
                "bug_type": bug
            }
        }
    except:
        return None

# --- EXECUTION ---
print(f"--- STARTING FACTORY ---")
print(f"Target: {TOTAL_SAMPLES} samples")
print(f"Model: {TEACHER_MODEL}")

success_count = 0
pbar = tqdm(total=TOTAL_SAMPLES, desc="Generating Data")

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    while success_count < TOTAL_SAMPLES:
        data = generate_sample()
        if data:
            f.write(json.dumps(data) + "\n")
            f.flush()
            success_count += 1
            pbar.update(1)

print(f"\n--- PROCESS COMPLETED ---")
print(f"Data saved to: {OUTPUT_FILE}")