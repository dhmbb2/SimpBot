from pdfminer.high_level import extract_text
# import nltk
# from nltk.tokenize import sent_tokenize
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.init_index()
    
    def init_index(self):
        print(f"Preparing indexing for knowledge Base")
        # get text from pdf
        text = self.extract_text_from_pdf()
        print(f"length of PDF: {len(text)}")

        # split the text into chunks
        self.chunks = self.split_text_into_chunks_by_characters(text)
        print(f"Number of chunks: {len(self.chunks)}")

        # get embedding with sentence transformer
        chunk_embeddings = self.model.encode(self.chunks)
        self.index = self.create_faiss_index(chunk_embeddings)
    

    def split_text_into_chunks_by_characters(self, text, chunk_size=500):
        text_length = len(text)
        chunks = []

        for i in range(0, text_length, chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks

    def extract_text_from_pdf(self):
        text = extract_text(self.pdf_path)
        return text

    def create_faiss_index(self, embeddings):
        embeddings_np = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)
        return index
        
    def search(self, query, top_k=3):
        """
        Search for the most similar top_k chunks in the database with the query.
        """
        query_embedding = self.model.encode([query])[0].astype('float32')
        D, I = self.index.search(np.array([query_embedding]), k=top_k)
        chunk_sim = []
        # we strip the first token and last token because they can be incomplete
        for idx in I[0]:
            chunk_sim.append(" ".join(self.chunks[idx].split()[1:-1]))
        return chunk_sim