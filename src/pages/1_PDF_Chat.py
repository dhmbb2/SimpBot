import streamlit as st
from pathlib import Path
from chatbot import ChatBot
from utils import Message
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import asyncio

path = Path("./articles")
file_names = [f.name for f in path.iterdir() if f.is_file()]
file_dict = {}
for name in file_names:
    file_dict[name] = path.joinpath(name)

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

if "page_state" not in st.session_state or st.session_state["page_state"] != "page_1":
    st.session_state["page_state"] = "page_1"
    st.session_state["messages"] = []

def update_bot_callbacks():
    st.session_state["messages"] = []
    with st.spinner("Analyzing the file, please wait a while..."):
        st.session_state["bot"] = ChatBot(
            st.session_state["model"], 
            st.session_state["tokenizer"], 
            st.session_state["device"],
            pdf_path=file_dict[st.session_state["pdf_choice"]]    
        )

with st.sidebar:
    pdf_choice = st.selectbox(
        "Please choose the PDF you want to do Q&A with.",
        file_names,
        key="pdf_choice",
        index=None,
        placeholder="Select a file...",
        on_change=update_bot_callbacks,
    )
    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        response = st.session_state["bot"].clear_history()


st.title("📝 File Q&A with SimpBot")
st.caption("Ask me anything about the document. 📚")

if pdf_choice is None:
    st.info("Please select a file to continue.")
else:
    print(pdf_choice)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    prompt = st.chat_input(
        "Ask something about the article"
    )

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
            # with st.spinner("Thinking..."):
            #     response = st.session_state["bot"].chat_with_bot(prompt)
            # st.session_state["messages"].append(Message(identity=1, message=response))
            # with st.chat_message("assistant"):
            #     st.markdown(response)
            # empty placeholder instead of thinking
            with st.chat_message("assistant"):
                response_placeholder = st.empty()

            async def generate_response():
                response_gen = st.session_state["bot"].async_chat_with_bot(prompt)
                # Update the UI dynamically
                async for response_part in response_gen:
                    response_placeholder.markdown(response_part)
                # Once the response is fully generated, add it to the chat history
                st.session_state["messages"].append(Message(identity=1, message=response_part))
                # with st.chat_message("assistant"):
                #     st.markdown(response_part)
                
            asyncio.run(generate_response())