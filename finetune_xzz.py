"""
The main program for finetuning LLMs with Huggingface Transformers Library.

ALL SECTIONS WHERE CODE POSSIBLY NEEDS TO BE FILLED IN ARE MARKED AS TODO.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
# from torch.nn import DataParallel
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets

# %%
# Define the arguments required for the main program.
# NOTE: You can customize any arguments you need to pass in.
@dataclass
class ModelArguments:
    """Arguments for model
    """
    model_name_or_path: Optional[str] = field(
        default="/data/youjunqi/nlp/qwen",
        metadata={
            "help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."
        }
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype."
            ),
            "choices": ["bfloat16", "float16", "float32"],
        },
    )
    max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    # TODO: add your model arguments here


@dataclass
class DataArguments:
    """Arguments for data"""
    dataset_path: Optional[str] = field(
        default="/data/youjunqi/nlp/alpaca",  # Using the Alpaca dataset for fine-tuning
        metadata={
            "help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."
        }
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """Arguments for training"""
    output_dir: str = field(default="./results/1epoch", metadata={"help": "Output directory for results."})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "Batch size for training"})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    save_steps: int = field(default=500, metadata={"help": "Steps interval for saving model"})
    logging_dir: str = field(default="./logs", metadata={"help": "Logging directory"})
    logging_steps: int = field(default=10, metadata={"help": "Interval for logging"})
    warmup_steps: int = field(default=500, metadata={"help": "Number of warmup steps"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay value"})
    save_total_limit: int = field(default=2, metadata={"help": "Limit number of saved checkpoints"})
    torch_precision: Optional[str] = field(default="bfloat16", metadata={"help": "Data type for model parameters"})  # 修改此处
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove columns not required by the model"})


# %%
# The main function
# NOTE You can customize some logs to monitor your program.
def finetune():
    # TODO Step 1: Define an arguments parser and parse the arguments
    # NOTE Three parts: model arguments, data arguments, and training arguments
    # HINT: Refer to 
    def parse_args():
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        return model_args, data_args, training_args
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/internal/trainer_utils#transformers.HfArgumentParser
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.TrainingArguments
    # parser = parse_args()
    model_args, data_args, training_args = parse_args()

    # TODO Step 2: Load tokenizer and model
    def load_model_and_tokenizer(model_args):
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        return tokenizer, model
    # HINT 1: Refer to
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/tokenizer#tokenizer
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/qwen2
    # HINT 2: To save training GPU memory, you need to set the model's parameter precision to half-precision (float16 or bfloat16).
    #         You may also check other strategies to save the memory!
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/llama2#usage-tips
    #   * https://huggingface.co/docs/transformers/perf_train_gpu_one
    #   * https://www.53ai.com/news/qianyanjishu/2024052494875.html
    tokenizer, model = load_model_and_tokenizer(model_args)

    # model = DataParallel(model)
    # TODO Step 3: Load dataset
    # HINT: https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset
    def load_dataset(data_args):
        dataset = datasets.load_dataset(data_args.dataset_path, split='train')
        return dataset
    
    dataset = load_dataset(data_args)
    # print(dataset["train"].column_names)


    # TODO Step 4: Define the data collator function
    # NOTE During training, for each model parameter update, we fetch a batch of data, perform a forward and backward pass,
    # and then update the model parameters. The role of the data collator is to process the data (e.g., padding the data within
    # a batch to the same length) and format the batch into the input required by the model.
    #
    # In this assignment, the purpose of the custom data_collator is to process each batch of data from the dataset loaded in
    # Step 3 into the format required by the model. This includes tasks such as tokenizing the data, converting each token into 
    # an ID sequence, applying padding, and preparing labels.
    # 
    # HINT:
    #   * Before implementation, you should:
    #      1. Clearly understand the format of each sample in the dataset loaded in Step 3.
    #      2. Understand the input format required by the model (https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2ForCausalLM).
    #         Reading its source code also helps!

    def data_collator(batch):
        """
        Processes the batch into the format suitable for training the Qwen2Model.
        Only handles input_ids and labels, other parameters are left to default behavior.
        """
        # Combine 'instruction' and 'input' to form the full input for the model
        inputs = ["<|im_start|>system\n" + item['instruction'] + "<|im_end|>\n<|im_start|>user\n" + item['input'] + "<|im_end|>\n<|im_start|>assistant\n" for item in batch]
        instruction_lens = [len(tokenizer.tokenize(input)) for input in inputs]
        inputs = ["<|im_start|>system\n" + item['instruction'] + "<|im_end|>\n<|im_start|>user\n" + item['input'] + "<|im_end|>\n<|im_start|>assistant\n" + item['output'] + "<|endoftext|>" for item in batch]
    
        max_length = model_args.max_length
        # Tokenize the inputs and targets with padding and truncation to ensure consistent length
        inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    
        # Prepare the input_ids and labels
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].clone()
        for i, inst_len in enumerate(instruction_lens):
            attention_mask[i, :inst_len] = 0
        labels = input_ids.clone()
        padding_token_id = tokenizer.pad_token_id
        labels[:,1:][labels[:,1:] == padding_token_id] = -100

    
        # Ensure both input_ids and labels are padded to the same length
        assert input_ids.shape[1] == labels.shape[1], f"Mismatch in input_ids and labels length: {input_ids.shape[1]} vs {labels.shape[1]}"
    
    
        # Return the batch in the correct format for the model
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # TODO Step 5: Define the Trainer
    # HINT: https://huggingface.co/docs/transformers/main_classes/trainer
    def train_model(model, tokenizer, training_args, dataset, data_collator):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,  # Assuming the dataset has "train" split
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model(training_args.output_dir)

    # Step 6: Train!
    train_model(model, tokenizer, training_args, dataset, data_collator)

# %%
# Pass your training arguments.
# NOTE [IMPORTANT!!!] DO NOT FORGET TO PASS PROPER ARGUMENTS TO SAVE YOUR CHECKPOINTS!!!
sys.argv = [
    "notebook",                              # 当前脚本名（通常为 'notebook'）
    "--model_name_or_path", "/data/youjunqi/nlp/qwen",  # 预训练模型路径
    "--dataset_path", "/data/youjunqi/nlp/alpaca",       # 使用 Alpaca 数据集进行微调
    "--output_dir", "./results/1epoch",              # 训练结果保存路径
    "--per_device_train_batch_size", "2",    # 每设备训练批次大小
    "--learning_rate", "2e-5",
    "--num_train_epochs", "1",                # 训练的轮数
    "--logging_dir", "./logs",                # 日志保存路径
    "--save_steps", "200",                    # 保存模型的步数
    "--save_total_limit", "2",                # 保留最近的 2 个检查点
    "--logging_steps", "10",                     
    "--warmup_ratio", "0.03",                  
    "--weight_decay", "0.01",                 
    "--torch_precision", "bfloat16", 
    "--gradient_accumulation_steps", "32",
]
finetune()