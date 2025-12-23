from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# --- CONFIGURATION ---
# Dosya yollarını senin proje yapına göre ayarladım
DATA_PATH = "../dataset/clean/clean_train_data.jsonl"
OUTPUT_DIR = "../models/github_agent_v1"
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"

# Hyperparameters for RTX 3070 (8GB VRAM)
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

def train():
    print(f"🚀 Training Pipeline Started...")
    print(f"📂 Dataset: {DATA_PATH}")

    # 1. Load Base Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none", 
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False, 
        loftq_config = None,
    )

    # 3. Prepare Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"📊 Training with {len(dataset)} samples.")

    # Prompt Template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token 

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Output zaten string formatında JSON olduğu için direkt ekliyoruz
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 4. Training Arguments
    print("🔥 Starting SFT Trainer...")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 1, # Safe for 8GB
            gradient_accumulation_steps = 4, # Effective batch size = 4
            warmup_steps = 10,
            # num_train_epochs = 1, # Tam 1 tur dönsün (Yaklaşık 115 adım)
            max_steps = 120, # Garanti olsun diye 120 adım diyelim (~1 epoch)
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit", 
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "checkpoints", # Geçici kayıtlar
            save_strategy = "no", # Yer kaplamasın diye checkpoint almıyoruz
        ),
    )

    # 5. Train & Save
    trainer_stats = trainer.train()
    
    print(f"💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save GGUF for Ollama (Optional - Step for later)
    # model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method = "q4_k_m")

    print("✅ Training Completed Successfully!")

if __name__ == "__main__":
    train()