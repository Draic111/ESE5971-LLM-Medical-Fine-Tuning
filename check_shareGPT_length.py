import numpy as np
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

# ================= Configuration =================
MODEL_PATH = "./models/qwen0.5b"
DATA_PATH = "./data/train_en_1_sharegpt.jsonl"
PLOT_OUTPUT_PATH = "length_distribution.png"

def is_valid_conversation(convs):
    """
    Decoupled Check 1: Null/Empty Validation.
    """
    if not convs or len(convs) == 0:
        return False
    
    for msg in convs:
        if not str(msg.get("value", "")).strip():
            return False
    return True

def is_duplicate(convs, seen_hashes):
    """
    Decoupled Check 2: Duplicate Detection.
    """
    conv_sig = json.dumps(convs, sort_keys=True)
    if conv_sig in seen_hashes:
        return True
    
    seen_hashes.add(conv_sig)
    return False

def calculate_token_length(convs, tokenizer):
    """
    Decoupled Check 3: Token Length Calculation (ChatML format).
    """
    input_ids = []
    
    # Simulate System Prompt
    sys_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    input_ids.extend(tokenizer.encode(sys_prompt, add_special_tokens=False))
    
    # Simulate Multi-turn Conversation
    for msg in convs:
        role = msg.get("from", "").lower()
        value = msg.get("value", "")
        
        if role in ["human", "user"]:
            text = f"<|im_start|>user\n{value}<|im_end|>\n"
        elif role in ["gpt", "assistant", "bot"]:
            text = f"<|im_start|>assistant\n{value}<|im_end|>\n"
        else:
            continue
            
        input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        
    return len(input_ids)

def plot_length_distribution(lengths):
    """
    Decoupled Task 4: Visualization.
    Generates a histogram of token lengths.
    """
    if not lengths:
        print("⚠️ No data available to plot.")
        return

    # Using standard matplotlib plotting
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Labeling the axes and title
    plt.xlabel('Token Length')
    plt.ylabel('Number of Samples (Frequency)')
    plt.title('Distribution of Token Lengths')
    
    # Adding grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    

def main():
    print(f"🔍 Loading Tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    
    print(f"📖 Loading Dataset: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    
    lengths = []
    seen_hashes = set()
    
    # Statistics counters
    total_initial = len(dataset)
    empty_count = 0
    duplicate_count = 0

    print("⏳ Starting decoupled preprocessing and analysis...")
    
    for example in dataset:
        convs = example.get("conversations", [])
        
        if not is_valid_conversation(convs):
            empty_count += 1
            continue
            
        if is_duplicate(convs, seen_hashes):
            duplicate_count += 1
            continue
            
        token_len = calculate_token_length(convs, tokenizer)
        lengths.append(token_len)

    # --- Generate Visualization ---
    plot_length_distribution(lengths, PLOT_OUTPUT_PATH)

    # ================= Final Report =================
    lengths_np = np.array(lengths)
    print("\n📊 =============== Data Analysis Report ===============")
    print(f"Initial Total:      {total_initial} samples")
    print(f"Empty/Invalid:      {empty_count} samples")
    print(f"Duplicates:         {duplicate_count} samples")
    print(f"Final Valid Total:  {len(lengths)} samples")
    print("-" * 45)
    if len(lengths_np) > 0:
        print(f"Min Length:         {np.min(lengths_np)} tokens")
        print(f"Max Length:         {np.max(lengths_np)} tokens")
        print(f"Mean Length:        {np.mean(lengths_np):.2f} tokens")
        print(f"95th Percentile:    {np.percentile(lengths_np, 95):.0f} tokens")
    print("======================================================\n")

if __name__ == "__main__":
    main()