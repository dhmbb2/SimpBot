from typing import List, Dict
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

# Define the arguments for the model, data, and training
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="./model/qwen3b",
        metadata={"help": "The path to the model to fine-tune or its name on the Hugging Face Hub."}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Precision type for model loading.", "choices": ["float16", "bfloat16", "float32"]}
    )

@dataclass
class DataArguments:
    dataset_path: str = field(
        default="./alpaca/",
        metadata={"help": "Path to the fine-tuning dataset or its name on Hugging Face Hub."}
    )
    max_input_length: int = field(
        default=2048,
        metadata={"help": "Maximum input sequence length."}
    )
    max_output_length: int = field(
        default=2048,
        metadata={"help": "Maximum output sequence length."}
    )

# Main function for fine-tuning
def finetune():

    # Step 1: Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Step 2: Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch.bfloat16
    )

    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_model_test_lora/",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        gradient_accumulation_steps=8,
        save_steps=400,
        bf16=True,
        logging_steps=20,
        report_to=['tensorboard']
    )


    # Step 3: Load dataset
    dataset = load_dataset(data_args.dataset_path, split='train')

    def preprocess_function(examples):
        inputs = [f"Instruction: {instruction} Input: {inp}" for instruction, inp in zip(examples["instruction"], examples["input"])]
        inputs_len = [len(tokenizer.tokenize(inp)) for inp in inputs]
        inputs = [f"{inp} Output: {outp}" for inp, outp in zip(inputs, examples["output"])]
        final_token_spot = [len(tokenizer.tokenize(inp)) for inp in inputs]


        inputs = tokenizer(
            inputs, truncation=True, padding="max_length", max_length=data_args.max_input_length, return_tensors="pt"
        )

        attention_mask = inputs["attention_mask"].clone()
        # for i, inst_len in enumerate(inputs_len):
        #     attention_mask[i, :inst_len] = 0
        labels = inputs["input_ids"].clone()
        padding_token_id = tokenizer.pad_token_id
        labels[labels == padding_token_id] = -100
        for i, fs in enumerate(final_token_spot):
            labels[i, fs] = tokenizer.eos_token_id
        for i, inst_len in enumerate(inputs_len):
            labels[i, :inst_len] = -100
        inputs["attention_mask"] = attention_mask
        inputs["labels"] = labels
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # # Step 4: Define the data collator
    def data_collator(batch: List[Dict]):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    # Step 5: Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # eval_dataset=dataset.get("validation"),  # Optional
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Step 6: Train the model
    trainer.train()

if __name__ == "__main__":
    finetune()

