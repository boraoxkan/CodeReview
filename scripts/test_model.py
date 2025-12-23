"""
Code Review Model Test Suite
-----------------------------
Bu script, fine-tune edilmis Code Review modelini kapsamli sekilde test eder.

Kullanim:
    python test_model.py                    # Tum testleri calistir
    python test_model.py --interactive      # Interaktif mod (kendi kodunuzu test edin)
    python test_model.py --quick            # Hizli test (sadece 2 ornek)
"""

from unsloth import FastLanguageModel
import torch
import json
import argparse
from typing import Optional

# --- CONFIGURATION ---
MODEL_PATH = "../models/github_agent_v1"
MAX_SEQ_LENGTH = 2048

# --- TEST CASES ---
# Farkli diller ve bug turleri icin test ornekleri
TEST_CASES = [
    {
        "name": "SQL Injection (Python)",
        "language": "Python",
        "bug_type": "SQL Injection",
        "code": """
def get_user(username):
    cursor = db.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()
"""
    },
    {
        "name": "XSS Vulnerability (JavaScript)",
        "language": "JavaScript",
        "bug_type": "XSS",
        "code": """
function displayMessage(userInput) {
    document.getElementById('output').innerHTML = userInput;
}
"""
    },
    {
        "name": "Command Injection (Python)",
        "language": "Python",
        "bug_type": "Command Injection",
        "code": """
import os

def ping_host(hostname):
    command = "ping -c 4 " + hostname
    os.system(command)
"""
    },
    {
        "name": "Hardcoded Credentials (Java)",
        "language": "Java",
        "bug_type": "Hardcoded Credentials",
        "code": """
public class DatabaseConnection {
    private static final String DB_USER = "admin";
    private static final String DB_PASSWORD = "super_secret_123";

    public Connection connect() {
        return DriverManager.getConnection(URL, DB_USER, DB_PASSWORD);
    }
}
"""
    },
    {
        "name": "Race Condition (Python)",
        "language": "Python",
        "bug_type": "Race Condition",
        "code": """
balance = 1000

def withdraw(amount):
    global balance
    if balance >= amount:
        # Time gap here allows race condition
        balance = balance - amount
        return True
    return False
"""
    },
    {
        "name": "Memory Leak (C++)",
        "language": "C++",
        "bug_type": "Memory Leak",
        "code": """
void processData() {
    int* data = new int[1000];
    // Process data...
    if (error_occurred) {
        return;  // Memory leak: data not freed
    }
    delete[] data;
}
"""
    },
    {
        "name": "Off-by-One Error (JavaScript)",
        "language": "JavaScript",
        "bug_type": "Off-by-One",
        "code": """
function getLastElements(arr, n) {
    let result = [];
    for (let i = arr.length; i > arr.length - n; i--) {
        result.push(arr[i]);  // arr[arr.length] is undefined
    }
    return result;
}
"""
    },
    {
        "name": "Null Pointer (Java)",
        "language": "Java",
        "bug_type": "Null Pointer",
        "code": """
public String getUserEmail(int userId) {
    User user = userRepository.findById(userId);
    return user.getEmail();  // user might be null
}
"""
    },
]


class CodeReviewTester:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze this {} code for defects.

### Input:
{}

### Response:
"""

    def load_model(self):
        """Modeli yukle"""
        print(f"\n{'='*60}")
        print(f"Model Yukleniyor: {self.model_path}")
        print(f"{'='*60}\n")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model basariyla yuklendi!\n")

    def analyze_code(self, code: str, language: str = "Python") -> dict:
        """Kod analizi yap"""
        prompt = self.alpaca_prompt.format(language, code)

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,
            temperature=0.1,
            do_sample=True,
        )

        response_text = self.tokenizer.batch_decode(outputs)[0]
        json_part = response_text.split("### Response:")[-1]
        # Qwen special tokens temizle
        json_part = json_part.replace("<|endoftext|>", "")
        json_part = json_part.replace("<|end_of_text|>", "")
        json_part = json_part.replace("<|im_end|>", "")
        json_part = json_part.replace("<|im_start|>", "")
        json_part = json_part.strip()

        # JSON parse etmeyi dene
        try:
            result = json.loads(json_part)
            return {"success": True, "data": result, "raw": json_part}
        except json.JSONDecodeError:
            return {"success": False, "data": None, "raw": json_part}

    def run_single_test(self, test_case: dict, index: int = 0) -> dict:
        """Tek bir test calistir"""
        print(f"\n{'='*60}")
        print(f"TEST #{index + 1}: {test_case['name']}")
        print(f"Dil: {test_case['language']} | Bug Turu: {test_case['bug_type']}")
        print(f"{'='*60}")
        print(f"\n--- KOD ---")
        print(test_case['code'])
        print(f"\n--- ANALIZ EDILIYOR... ---\n")

        result = self.analyze_code(test_case['code'], test_case['language'])

        # Her zaman raw ciktiyi goster
        print("--- RAW OUTPUT ---")
        print(result['raw'][:1500] if len(result['raw']) > 1500 else result['raw'])
        print("--- RAW OUTPUT END ---\n")

        if result['success']:
            data = result['data']
            print("--- PARSED JSON ---\n")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Ozet goster
            if isinstance(data, dict):
                print(f"\n--- OZET ---")
                if 'code_quality_score' in data:
                    score = data['code_quality_score']
                    print(f"Kod Kalite Skoru: {score}/100")
                if 'critical_issues' in data:
                    issues = data['critical_issues']
                    print(f"Kritik Sorun Sayisi: {len(issues) if isinstance(issues, list) else 'N/A'}")
        else:
            print("[!] JSON parse edilemedi")

        return result

    def run_all_tests(self, quick: bool = False):
        """Tum testleri calistir"""
        if self.model is None:
            self.load_model()

        test_cases = TEST_CASES[:2] if quick else TEST_CASES
        results = []

        print(f"\n{'#'*60}")
        print(f"  CODE REVIEW MODEL TEST SUITE")
        print(f"  Toplam Test: {len(test_cases)}")
        print(f"{'#'*60}")

        for i, test_case in enumerate(test_cases):
            result = self.run_single_test(test_case, i)
            results.append({
                "test_name": test_case['name'],
                "success": result['success'],
                "bug_type": test_case['bug_type']
            })

        # Sonuc ozeti
        print(f"\n{'#'*60}")
        print(f"  TEST SONUCLARI")
        print(f"{'#'*60}\n")

        success_count = sum(1 for r in results if r['success'])
        print(f"Basarili JSON Parse: {success_count}/{len(results)}")
        print(f"\nDetaylar:")
        for r in results:
            status = "[OK]" if r['success'] else "[RAW]"
            print(f"  {status} {r['test_name']}")

        return results

    def interactive_mode(self):
        """Interaktif mod - kullanici kendi kodunu test edebilir"""
        if self.model is None:
            self.load_model()

        print(f"\n{'#'*60}")
        print(f"  INTERAKTIF TEST MODU")
        print(f"  Cikmak icin 'exit' veya 'quit' yazin")
        print(f"{'#'*60}\n")

        while True:
            print("\nProgramlama dili (Python/JavaScript/Java/C++/Go/etc.):")
            language = input("> ").strip()

            if language.lower() in ['exit', 'quit', 'q']:
                print("Cikiliyor...")
                break

            if not language:
                language = "Python"
                print(f"(Varsayilan: {language})")

            print(f"\nAnaliz edilecek kodu girin (bos satir + ENTER ile bitirin):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            code = "\n".join(lines)

            if not code.strip():
                print("Kod girilmedi, tekrar deneyin.")
                continue

            print(f"\n{'='*60}")
            print("ANALIZ EDILIYOR...")
            print(f"{'='*60}")

            result = self.analyze_code(code, language)

            if result['success']:
                print("\n--- ANALIZ SONUCU ---\n")
                print(json.dumps(result['data'], indent=2, ensure_ascii=False))
            else:
                print("\n--- MODEL CIKTISI ---\n")
                print(result['raw'])


def main():
    parser = argparse.ArgumentParser(description="Code Review Model Test Suite")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interaktif mod - kendi kodunuzu test edin")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Hizli test - sadece 2 ornek")
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH,
                        help=f"Model yolu (varsayilan: {MODEL_PATH})")

    args = parser.parse_args()

    tester = CodeReviewTester(model_path=args.model)

    if args.interactive:
        tester.interactive_mode()
    else:
        tester.run_all_tests(quick=args.quick)


if __name__ == "__main__":
    main()
