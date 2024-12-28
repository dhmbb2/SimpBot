from typing import List, Dict
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import argparse


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="./model/qwen",
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
    lossmask: bool = field(
        default=True,
        metadata={"help": "whether or not use the mask loss"}
    )

# Main function for fine-tuning
def finetune():

    # Parsing arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch.bfloat16
    )

    training_args = TrainingArguments(
        output_dir="./fine_tuned_model_test2/",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        gradient_accumulation_steps=8,
        save_steps=400,
        bf16=True,
        logging_steps=20,
        report_to=['tensorboard']
    )


    # Loading dataset
    dataset = load_dataset(data_args.dataset_path, split='train')

    # preprocess the data to accelerate training
    def preprocess_function(examples):
        inputs = [f"Instruction: {instruction} Input: {inp}" for instruction, inp in zip(examples["instruction"], examples["input"])]
        inputs_len = [len(tokenizer.tokenize(inp)) for inp in inputs]
        inputs = [f"{inp} Output: {outp}" for inp, outp in zip(inputs, examples["output"])]
        final_token_spot = [len(tokenizer.tokenize(inp)) for inp in inputs]

        inputs = tokenizer(
            inputs, truncation=True, padding="max_length", max_length=data_args.max_input_length, return_tensors="pt"
        )

        attention_mask = inputs["attention_mask"].clone()
        labels = inputs["input_ids"].clone()
        padding_token_id = tokenizer.pad_token_id
        labels[labels == padding_token_id] = -100
        # pad_token_id and eos is the same for Qwen, we 
        # restore the eos to prevent repeating token generation
        if data_args.lossmask:
            for i, fs in enumerate(final_token_spot):
                labels[i, fs] = tokenizer.eos_token_id
        for i, inst_len in enumerate(inputs_len):
            labels[i, :inst_len] = -100
        inputs["attention_mask"] = attention_mask
        inputs["labels"] = labels
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    def data_collator(batch: List[Dict]):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == "__main__":
    finetune()

