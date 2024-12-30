import torch
from typing import List, Dict
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from peft import PeftModel

from character import Character
from rag import KnowledgeBase


class ChatBot:
    def __init__(
        self,
        model, 
        tokenizer, 
        device="cpu", 
        max_history=5, 
        max_length=150, 
        temperature=0.7, 
        top_p=0.9, 
        top_k=50, 
        pdf_path=None,
        character_path=None,
        chara_name=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.history = [] 
        self.device = device
        self.max_history = max_history 
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        self.knowledge_base = None
        if pdf_path is not None:
            self.knowledge_base = KnowledgeBase(pdf_path)
        self.character = None
        if character_path is not None:
            assert chara_name is not None, "please provide the character name"
            self.character = Character(character_path, chara_name)


    def _get_history_context(self, in_chat_form=False):
        # get history message as context
        context = ""
        if in_chat_form:
            for instruction, response in self.history[-self.max_history:]: 
                context += f"User:{instruction}\n Bot:{response}\n"
        else:
            for instruction, response in self.history[-self.max_history:]: 
                context += f"{instruction}\n {response}\n"
        return context

    def _update_history(self, user_input, bot_response):
        """更新历史对话"""
        self.history.append((user_input, bot_response))
        # 保证历史对话不超过 max_history 条
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def clean_response(self, response):
        """
        Clean all sorts of quotes.
        """
        if response.startswith("\"") or response.startswith("\'") or \
            response.startswith("“") or response.startswith("「"):
            response = response[1:]
        if response.endswith("\"") or response.endswith("\'") or \
            response.endswith("”") or response.endswith("」"):
            response = response[:-1]
        return response

    def clear_history(self):
        self.history = [] 

    def chat_with_bot(self, user_input):
        
        if self.character is not None:
            history = self._get_history_context(in_chat_form=False)
            prompt = self.character.get_input_prompt(user_input, history)
        else:
            history = self._get_history_context(in_chat_form=True)
            context = ""
            if self.knowledge_base is not None:
                context = " ".join(self.knowledge_base.search(user_input))
            
            # obtain input string via pretrained Instruction format
            prompt = f"You are a chatting bot, please respond to the following user input base on the provided input information. \
                    Instruction: {context} {history} User: {user_input} Output: Bot:"

        print(f"history length: {len(self.history)}")
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # trim the context and history if the embedding length exceeds the configed maximum length
        if len(inputs[0]) > (self.model.config.max_position_embeddings):
            res = len(inputs[0]) - (self.model.config.max_position_embeddings)
            # we first try giving up context
            context = ' '.join(context.split()[:-res])
            res -= len(inputs[0]) - len(context.split())
            # if not enough, we trim the history
            history = " ".join(context.split()[:-res])

            # reformulate input
            prompt = f"You are a chatting bot, please respond to the following user input base on the provided input information. \
                    Instruction: {context} {history} User: {user_input} Output: "
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        print(f"Input ID's length: {len(inputs[0])}.")
        # inputs = inputs[:, -self.model.config.max_position_embeddings:]

        # 生成模型响应
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                # max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解码生成的响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # strip the context part 
        bot_response = response[len(prompt):].strip()
        # clean any possible quotes
        bot_response = self.clean_response(bot_response)

        # 更新历史对话
        self._update_history(user_input, bot_response)

        return bot_response

# 聊天函数
def start_chat():
    # model_path = "/data/youjunqi/nlp/fine_tuned_model_test2/checkpoint-1600"  # path to my finetuned model
    model_path = "./qwen3b"
    lora_ckpt_path = "./fine_tuned_model_test_lora/checkpoint-9600"
    # pdf_path = "/data/youjunqi/nlp/articles/civilcode.pdf"
    # character_path = "./Sheldon.jsonl"
    pdf_path = None
    device="cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model = PeftModel.from_pretrained(model, lora_ckpt_path)
    chatbot = ChatBot(model, tokenizer, device, pdf_path=pdf_path, character_path=character_path, chara_name="Sheldon")
    print("Chatbot: Hello! How can I help you today? (Type 'quit' to exit)")

    while True:
        # 获取用户输入
        user_input = input("You: ")
        
        # 如果用户输入'quit'，退出聊天
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        
        # 获取聊天机器人响应
        response = chatbot.chat_with_bot(user_input)
        
        # 打印聊天机器人响应
        print(f"Chatbot: {response}")


if __name__ == "__main__":

    start_chat()