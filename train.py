import os
import torch
from typing import Optional

from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, TaskType, get_peft_model

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        metadata={"help": "Local model directory"}
    )

    data_path: str = field(
        metadata={"help": "Local dataset directory"}
    )

    max_train_samples: Optional[int] = field(
        default = None,
        metadata = {"help": "Number of data you want to use for training"}
    )

    use_qlora: bool = field(
        default=False, 
        metadata={"help": "Whether using 4-bit QLoRA"}
    )

    lora_rank: int = field(
        default=16, 
        metadata={"help": "Rank of the LoRA"}
    )

    lora_alpha: int = field(
        default=32, 
        metadata={"help": "LoRA alpha parameter"}
    )

    lora_dropout: float = field(
        default = 0.05,
        metadata = {"help": "LoRA dropout probability"}
    )

    max_seq_length: int = field(
        default=1024, 
        metadata={"help": "Maximum model context length. Suggested: 8192, 4096, 2048, 1024, 512"}
    )



def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_json_file("config.json")

    set_seed(training_args.seed)
    IGNORE_INDEX = -100 # Ignore non-answering part when calculating loss

    print(f"Initializing Training | Model: {script_args.model_name_or_path}")

    # ================= Loading Tokenizer (Qwen + ChatML) =================
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False)
    
    # EOS token: <|im_end|>
    # Find id of pad token 
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|im_end|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if tokenizer.pad_token_id is None: # backtrack to eos_token if id not obtained
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
    tokenizer.padding_side = "right"

    # ================= Preprocessing ShareGPT Dataset =================
    full_dataset = load_dataset("json", data_files=script_args.data_path)["train"]

    # Triggering Shuffle Logic when max_train_samples is set
    if getattr(script_args, "max_train_samples", None) is not None:
        if len(full_dataset) > script_args.max_train_samples:
            print(f"Original sample amount ({len(full_dataset)}) excess limit ({script_args.max_train_samples})")
            print("Shuffling samples...")
            full_dataset = full_dataset.shuffle(seed=42).select(range(script_args.max_train_samples))
            print(f"Sampling finished! Now we have: {len(full_dataset)} samples.")

    # Train Test Split
    split_dataset = full_dataset.train_test_split(test_size = 0.1, seed = 42)
    raw_train = split_dataset["train"]
    raw_eval = split_dataset["test"]
    
    def preprocess_sharegpt(examples):
        input_ids_list, labels_list = [], []
        
        for conversations in examples["conversations"]:
            input_ids, labels = [], []
            
            # Plug in system prompt
            sys_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            sys_ids = tokenizer.encode(sys_prompt, add_special_tokens=False)
            input_ids.extend(sys_ids)
            labels.extend([IGNORE_INDEX] * len(sys_ids)) # System Prompt not participate in Loss calculation

            # Iterate 'conversations' among the dataset
            for msg in conversations:
                role = msg.get("from", "")
                value = msg.get("value", "")

                if role in ["human", "user"]:
                    text = f"<|im_start|>user\n{value}<|im_end|>\n"
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    input_ids.extend(ids)
                    labels.extend([IGNORE_INDEX] * len(ids))

                elif role in ["assistant", "gpt"]:
                    text = f"<|im_start|>assistant\n{value}<|im_end|>\n"
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    input_ids.extend(ids)
                    labels.extend(ids)

            # Truncating Logic
            if len(input_ids) > script_args.max_seq_length:
                input_ids = input_ids[:script_args.max_seq_length]
                labels = labels[:script_args.max_seq_length]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    print("Analysing ShareGPT dataset...")
    tokenized_train = raw_train.map(
        preprocess_sharegpt, 
        batched=True, 
        remove_columns=raw_train.column_names,
        desc="Tokenizing ShareGPT"
    )

    tokenized_eval = raw_eval.map(
        preprocess_sharegpt,
        batched = True,
        remove_columns=raw_eval.column_names   
    )

    # ================= Load Model =================
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16
    )
    model.config.use_cache = False 

    # ================= Load LoRA =================
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        r = script_args.lora_rank,
        lora_alpha = script_args.lora_alpha,
        lora_dropout = script_args.lora_dropout,
        target_modules = target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ================= Train =================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset = tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Start Training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)

    print(f"Training Completed")

if __name__ == "__main__":
    main()