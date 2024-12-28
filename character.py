import pandas as pd
from pdfminer.high_level import extract_text
# import nltk
# from nltk.tokenize import sent_tokenize
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Character:
    def __init__(self, character_file, chara_name):
        self.character_file = character_file
        self.chara_name = chara_name
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # self.sys_prompt = "I want you to act like Sheldon Cooper from Big Bang Theory.\nIf others‘ questions are related with the novel, please try to reuse the original lines from the novel. But don't include the person saying the line.\nI want you to respond and answer like Sheldon using the tone, manner and vocabulary Sheldon would use.\nYou must know all of the knowledge of Sheldon.\n\nSheldon has some social difficulties and sometimes displays awkward and inappropriate behavior. He likes to plan his life strictly according to his habits and schedule, not allowing any disruption to his routine. In front of friends, he often appears arrogant and self-righteous, believing that his intelligence is superior to others.\n"
        self.parse_character_file()

    def parse_character_file(self):
        df = pd.read_json(self.character_file, lines=True)
        df_dropped = df.drop(['luotuo_openai', "bge_zh_s15"], axis=1)
        self.sys_prompt = df_dropped.loc[0, "text"]
        df_dropped = df_dropped.drop([0, 1], axis=0)
        self.chunks = df_dropped["text"].to_list()
        
        # we let each text line to be a chunk and vectorized it
        chunk_embeddings = self.model.encode(self.chunks)
        self.index = self.create_faiss_index(chunk_embeddings)

    def create_faiss_index(self, embeddings):
        embeddings_np = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)
        return index

    def get_input_prompt(self, user_input, history, top_k=3):
        query_embedding = self.model.encode([user_input])[0].astype('float32')
        D, I = self.index.search(np.array([query_embedding]), k=top_k)
        chunk_sim = [self.chunks[idx] for idx in I[0]]
        # print(chunk_sim)
        prompt = f"{self.sys_prompt} The following script are the context you can reference from: {chunk_sim} \
                    The chat history is {history}. Only reply one line, and then stop. Don't copy what other characters says in the script.\
                    User: {user_input} Output: {self.chara_name}:"
        return prompt

if __name__ == "__main__":
    character_file = "/data/youjunqi/nlp/Sheldon.jsonl"
    chara = Character(character_file)
    user_input = "Hello!"
    print(chara.get_input_prompt(user_input, ""))