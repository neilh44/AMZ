from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
st.set_page_config(page_title="DocGenius: Document Generation AI")
st.header("Ask Your PDF📄")
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )  

    chunks = text_splitter.split_text(text)

    query = st.text_input("Ask your Question about your PDF")
    if query:
        tokenizer = AutoTokenizer.from_pretrained("mistral-community/Mixtral-8x22B-v0.1")
        model = AutoModelForCausalLM.from_pretrained("mistral-community/Mixtral-8x22B-v0.1")

        inputs = tokenizer(query, chunks, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=50)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.success(answer)
