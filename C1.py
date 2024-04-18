from dotenv import load_dotenv
import streamlit as st
import time
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.gpt2 import GPT2Embeddings  # Import GPT2 embeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # Import GPT-2 model
from langchain.callbacks import get_gpt2_callback  # Import GPT-2 callback

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
      
      # create embeddings
      embeddings = GPT2Embeddings()  # Use GPT-2 embeddings
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      with st.chat_message("user"):
        st.write("Hello World 👋")
      user_question = st.text_input("Please ask a question about your PDF here:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        # Load GPT-2 model
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Run inference with GPT-2 model
        with get_gpt2_callback() as cb:
          # Your inference code here
          st.write("Inference with GPT-2 model")
           
        # Display response
        st.write(response)

if __name__ == '__main__':
    main()
