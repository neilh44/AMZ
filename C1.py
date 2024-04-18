import torch
import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load Hugging Face model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load CSV data
github_raw_url = "https://github.com/neilh44/AMZ/raw/main/A1C2.csv"
df = pd.read_csv(github_raw_url)

def generate_order_ids(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7, attention_mask=input_ids)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    sku = input_text.split("'")[1]
    filtered_df = df[df['Sku'] == sku]
    order_ids = filtered_df['order id'].tolist()
    return generated_text, order_ids

st.title("Order ID Lookup App")

input_text = st.text_input("Enter input text", "provide order id with sku '2D-7N1V-0ZCB'")

if st.button("Generate"):
    generated_text, order_ids = generate_order_ids(input_text)
    st.write("Generated Text:", generated_text)
    st.write("Order IDs:", order_ids)
