import streamlit as st
from pathlib import Path

path = Path("./articles")
file_names = [f.name for f in path.iterdir() if f.is_file()]

with st.sidebar:
    pdf_choice = st.selectbox(
        "Please choose the PDF you want to do Q&A with.",
        file_names,
    )

st.title("📝 File Q&A with Anthropic")
question = st.text_input(
    "Ask something about the article",
    placeholder="Tell me something about ...",
)

# if uploaded_file and question and not anthropic_api_key:
#     st.info("Please add your Anthropic API key to continue.")

# if uploaded_file and question and anthropic_api_key:
#     article = uploaded_file.read().decode()
#     prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
#     {article}\n\n</article>\n\n{question}{anthropic.AI_PROMPT}"""

#     client = anthropic.Client(api_key=anthropic_api_key)
#     response = client.completions.create(
#         prompt=prompt,
#         stop_sequences=[anthropic.HUMAN_PROMPT],
#         model="claude-v1",  # "claude-2" for Claude 2 model
#         max_tokens_to_sample=100,
#     )
#     st.write("### Answer")
#     st.write(response.completion)