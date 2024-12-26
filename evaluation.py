from typing import List, Dict
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/data/youjunqi/nlp/qwen",
        metadata={"help": "The path to the model to fine-tune or its name on the Hugging Face Hub."}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Precision type for model loading.", "choices": ["float16", "bfloat16", "float32"]}
    )

@dataclass
class DataArguments:
    dataset_path: str = field(
        default="/data/youjunqi/nlp/alpaca/",
        metadata={"help": "Path to the fine-tuning dataset or its name on Hugging Face Hub."}
    )
    max_input_length: int = field(
        default=1024,
        metadata={"help": "Maximum input sequence length."}
    )
    max_output_length: int = field(
        default=1024,
        metadata={"help": "Maximum output sequence length."}
    )


if __name__=='__main__':

    device = "cuda"
    base_dir = '/data/youjunqi/nlp/qwen'
    womask ='/data/youjunqi/nlp/fine_tuned_model_test_no_mask/checkpoint-500'
    wmask = "/data/youjunqi/nlp/fine_tuned_model_test_with_mask/checkpoint-800"
    aat = "/data/youjunqi/nlp/fine_tuned_model_test2/checkpoint-1600"
    ##### fine-tuned
    tokenizer = AutoTokenizer.from_pretrained(aat, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(aat)
    model.to(device)
    input_seqs = ["Instruction: Based on the information provided, rewrite the sentence by changing its tense from past to future. Input: She played the piano beautifully for hours and then stopped as it was midnight."]

    inputs = tokenizer(input_seqs, padding='max_length',return_tensors="pt",max_length = 2048)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated_ids = model.generate(
        **inputs, 
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=2048,
        top_p=0.95,
        temperature=0.9,
        do_sample=True
    )

    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(generated_text)