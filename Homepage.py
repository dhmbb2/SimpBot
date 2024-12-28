import streamlit as st
import torch
from chatbot import ChatBot
from utils import Message
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# model and ckpt paths
model_path = "./qwen3b"
lora_ckpt_path = "/data/youjunqi/nlp/fine_tuned_model_test_lora/checkpoint-9600"
device="cuda" if torch.cuda.is_available() else "cpu"

# initialize LLM
if "model" not in st.session_state:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model = PeftModel.from_pretrained(model, lora_ckpt_path)
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer
    # Initialize the ChatBot
    print("Initializaing the Bot.")
    st.session_state["bot"] = ChatBot(st.session_state["model"], st.session_state["tokenizer"], device)

# Initialize some meta-data
st.set_page_config(page_title="Welcome to SimpBot")
st.title("🤠 Welcome to SimpBot")\
# st.caption("Type \"\\newsession\" to clear chat history and start a new chatting session.")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.container():
    st.header("Chat with SimpBot")

    for message in st.session_state["messages"]:
        if message.identity == 0:
            with st.chat_message("user"):
                st.markdown(message.message)
        elif message.identity == 1:
            with st.chat_message("assistant"):
                st.markdown(message.message)
    prompt = st.chat_input("Type something...")
    if prompt:
        st.session_state["messages"].append(Message(identity=0, message=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking..."):
            response = st.session_state["bot"].chat_with_bot(prompt)
        st.session_state["messages"].append(Message(identity=1, message=response))
        with st.chat_message("assistant"):
            st.markdown(response)