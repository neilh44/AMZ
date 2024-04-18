from dotenv import load_dotenv
import torch 
import streamlit as st
import time
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from transformers import GPT2Tokenizer, GPT2Model  # Import GPT-2 model and tokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import faiss

# Assuming `embeddings` is a list of numpy arrays representing vectors
def create_faiss_index(embeddings):
    # Convert embeddings to numpy array
    embeddings_np = np.vstack(embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    return index

# Usage
knowledge_base = create_faiss_index(embeddings)


# Sidebar contents
with st.sidebar:
    st.title('💬PDF Summarizer and Q/A App')
    st.markdown('''
    ## About this application
    You can built your own customized LLM-powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html) model
    ''')
    add_vertical_space(2)
    st.write(' Why drown in papers when your chat buddy can give you the highlights and summary? Happy Reading. ')
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
        
        # Create embeddings
        embeddings = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
      
        # Create FAISS index
        knowledge_base = FAISS.from_vectors(embeddings)
      
        # show user input
        with st.chat_message("user"):
            st.write("Hello World 👋")
        user_question = st.text_input("Please ask a question about your PDF here:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
        
            # Run inference with LangChain
            chain = load_qa_chain(OpenAI(), chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
           
            st.write(response)

if __name__ == '__main__':
    main()
