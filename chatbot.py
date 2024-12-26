import torch

class ChatBot:
    def __init__(self, model, tokenizer, device="cpu", max_history=5, max_length=150, temperature=0.7, top_p=0.9, top_k=50):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []  # 保存历史对话
        self.device = device
        self.max_history = max_history  # 限制保留的历史对话数量
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def _get_history_context(self):
        """获取历史对话的上下文"""
        context = ""
        # 遍历历史对话列表，拼接成字符串
        for instruction, response in self.history[-self.max_history:]:  # 保留最近的 `max_history` 对话
            context += f"User: {instruction}\nBot: {response}\n"
        return context

    def _update_history(self, user_input, bot_response):
        """更新历史对话"""
        self.history.append((user_input, bot_response))
        # 保证历史对话不超过 max_history 条
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def chat_with_bot(self, user_input):
        # 获取当前的对话上下文（包括历史对话）
        context = self._get_history_context()
        
        # 拼接用户输入和上下文
        prompt = context + f"User: {user_input}\nBot:"
        
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs[:, -self.model.config.max_position_embeddings:]  # 限制输入的最大长度

        # 生成模型响应
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解码生成的响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 获取机器人的响应文本（去除前面的上下文部分）
        bot_response = response[len(prompt):].strip()

        # 更新历史对话
        self._update_history(user_input, bot_response)

        return bot_response

# 聊天函数
def start_chat():
    model_path = "/data/youjunqi/nlp/fine_tuned_model_test2/checkpoint-1600"  # path to my finetuned model
    device="cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    chatbot = ChatBot(model, tokenizer, device)
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

    start_chat(model, tokenizer, device)