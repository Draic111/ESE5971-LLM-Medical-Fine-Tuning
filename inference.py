import argparse
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel

parser = argparse.ArgumentParser()

parser.add_argument(
    "--base_model", 
    type = str, 
    default = "./models/qwen2.5-7b",
)

parser.add_argument(
    "--lora_dir", 
    type=str, 
    default=None, 
)
args = parser.parse_args()

# Using 4-bit Quant to save VRAM
print("Loading with 4-bit Quantization for GPU acceleration")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    quantization_config=bnb_config, 
    device_map="auto"              
)


# LoRA Adapter Loading
if args.lora_dir:
    model = PeftModel.from_pretrained(model, args.lora_dir)
    print(f"LoRA adapter: {type(model)} loaded successfully")
else:
    print("No LoRA adapter specified, using base model for inference")

model.eval()


def chat():
    # Initialize the streamer. skip_prompt=True prevents the question from being re-printed.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Initializing history list
    history = [
        {"role": "system", "content": "You are a professional medical assistant."}
    ]
    # Max output token length
    MAX_CONTEXT_TOKENS = 2048

    while True:
        user_input = input("\n User: ")
        if user_input.strip().lower() == "exit": 
            break
        history.append({"role": "user", "content": user_input})

        if len(history) > 11: 
            history = [history[0]] + history[-10:] 
            
        # Automatically formatting history messages and add to history
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_CONTEXT_TOKENS
        ).to(model.device)
        
        # Configure Generation Parameters
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=512,       
            temperature=0.7,          
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"), 
            streamer=streamer       
        )
        
        # Threaded Execution for Typewriter Effect
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        print("Assistant: ", end="", flush=True)
        
        # Collecting model responses for updating memory
        full_response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_response += new_text
        print()

        # Adding to memory
        history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    chat()