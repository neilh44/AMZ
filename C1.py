from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

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

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask your Question about your PDF")
    if query:
        docs = knowledge_base.similarity_search(query)

        tokenizer = AutoTokenizer.from_pretrained("tuner007/mistral-7b")
        model = AutoModelForQuestionAnswering.from_pretrained("tuner007/mistral-7b")

        inputs = tokenizer(query, chunks, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        answer_tokens = all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        st.success(answer)
