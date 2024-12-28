import streamlit as st
from pathlib import Path
from chatbot import ChatBot
from utils import Message
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

path = Path("./chara")
cha_names = [f.stem for f in path.iterdir() if f.is_file()]
cha_dict = {}
for name in cha_names:
    cha_dict[name] = path.joinpath(name+".jsonl")

model_path = "./model/qwen3b"
lora_ckpt_path = "./model/qwen_lora"
device="cuda" if torch.cuda.is_available() else "cpu"

if "model" not in st.session_state or "tokenizer" not in st.session_state:
    with st.spinner("Initializing the LLM. Please wait a while."):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model = PeftModel.from_pretrained(model, lora_ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        st.session_state["model"] = model
        st.session_state["tokenizer"] = tokenizer
        st.session_state["device"] = device
if "page_state" not in st.session_state or st.session_state["page_state"] != "page_2":
    st.session_state["page_state"] = "page_2"
    st.session_state["messages"] = []

def update_bot_callbacks():
    with st.spinner("Building the character, please wait a while..."):
        st.session_state["messages"] = []
        st.session_state["bot"] = ChatBot(
            st.session_state["model"], 
            st.session_state["tokenizer"], 
            st.session_state["device"],
            character_path=cha_dict[st.session_state["cha_choice"]],
            chara_name=st.session_state["cha_choice"]
        )

with st.sidebar:
    cha_choice = st.selectbox(
        "Please choose the PDF you want to do Q&A with.",
        cha_names,
        key="cha_choice",
        index=None,
        placeholder="Select a file...",
        on_change=update_bot_callbacks,
    )
    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        response = st.session_state["bot"].clear_history()


st.title("🧙 Role-Playing with SimpBot")
st.caption("The performance is the best when using the same language with the character's name!")

if cha_choice is None:
    st.info("Please select a character to continue.")
else:
    print(cha_choice)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    prompt = st.chat_input(f"Have a chat with {cha_choice}...")

    with st.container():
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