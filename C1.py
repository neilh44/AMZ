from dotenv import load_dotenv
import streamlit as st
import time
from PyPDF2 import PdfReader, PdfReadError  # Import PdfReader and PdfReadError from PyPDF2
from streamlit_extras.add_vertical_space import add_vertical_space
from transformers import GPT2Tokenizer, GPT2Model  # Import GPT-2 model and tokenizer
from langchain.text_splitter import CharacterTextSplitter
import torch
import numpy as np
import faiss  # Import FAISS library

# Sidebar contents
with st.sidebar:
    st.title('💬PDF Summarizer and Q/A App')
    st.markdown('''
    ## About this application
    You can build your own customized LLM-powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html) model
    ''')
    add_vertical_space(2)
    st.write('Why drown in papers when your chat buddy can give you the highlights and summary? Happy Reading.')
    add_vertical_space(2)
    st.write('Made by ***Sangita Pokhrel***')

def main():
    load_dotenv()

    #Main Content
    st.header("Ask About Your PDF 🤷‍♀️💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF File and Ask Questions", type="pdf")
    
    # extract the text
    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
          
            # Load GPT-2 tokenizer and model
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2Model.from_pretrained("gpt2")
            
            # Generate embeddings for each chunk
            embeddings = []
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use the mean of the last hidden states as the embedding
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
          
            # Create FAISS index
            index = create_faiss_index(embeddings)
          
            # show user input
            with st.chat_message("user"):
                st.write("Hello World 👋")
            user_question = st.text_input("Please ask a question about your PDF here:")
            if user_question:
                # Perform similarity search with FAISS index
                query_embedding = compute_embedding(user_question, tokenizer, model)
                top_k = search_faiss_index(index, query_embedding, k=5)
                
                # Display top-k results
                st.write("Top 5 similar chunks:")
                for idx in top_k:
                    st.write(chunks[idx])
        except PdfReadError as e:
            st.error("Error reading the PDF file. Make sure the file is not encrypted or provide the password.")
            st.error(str(e))

def create_faiss_index(embeddings):
    # Convert embeddings to numpy array
    embeddings_np = np.vstack(embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    return index

def compute_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def search_faiss_index(index, query_embedding, k=5):
    distances, indices = index.search(np.expand_dims(query_embedding, axis=0), k)
    return indices.squeeze()

if __name__ == '__main__':
    main()
