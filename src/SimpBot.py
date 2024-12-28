import streamlit as st
import torch
from chatbot import ChatBot
from utils import Message
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# model and ckpt paths
model_path = "./model/qwen3b"
lora_ckpt_path = "./model/qwen_lora"
device="cuda" if torch.cuda.is_available() else "cpu"

# Initialize some meta-data
st.set_page_config(page_title="Welcome to SimpBot")
st.title("🤖 Welcome to SimpBot")
st.caption("Chat with me ❤️ !! Clike the new session button in the side bar to start a new session.")

# initialize LLM
if "model" not in st.session_state or "tokenizer" not in st.session_state:
    with st.spinner("Initializing the LLM. Please wait a while."):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model = PeftModel.from_pretrained(model, lora_ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        st.session_state["model"] = model
        st.session_state["tokenizer"] = tokenizer
        st.session_state["device"] = device

# if we change from other page to the home page, we will need to reinitialize the bot
if "page_state" not in st.session_state or st.session_state["page_state"] != "home":
    # Initialize the ChatBot
    st.session_state["bot"] = ChatBot(st.session_state["model"], st.session_state["tokenizer"], st.session_state["device"])
    st.session_state["page_state"] = "home"
    st.session_state["messages"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = st.chat_input("Type something...")

with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        response = st.session_state["bot"].clear_history()

with st.container():
    # st.header("Chat with SimpBot")

    for message in st.session_state["messages"]:
        if message.identity == 0:
            with st.chat_message("user"):
                st.markdown(message.message)
        elif message.identity == 1:
            with st.chat_message("assistant"):
                st.markdown(message.message)
    if prompt:
        st.session_state["messages"].append(Message(identity=0, message=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking..."):
            response = st.session_state["bot"].chat_with_bot(prompt)
        st.session_state["messages"].append(Message(identity=1, message=response))
        with st.chat_message("assistant"):
            st.markdown(response)