import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token_id to eos_token_id for open-end generation
model.config.pad_token_id = model.config.eos_token_id

# Initialize Llama Index components
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.2"
SYSTEM_PROMPT = """You are an AI assistant that analyzes data from the provided CSV file. Here are some rules you always follow:
- Generate human-readable output.
- Generate only the requested output, avoiding unnecessary information.
- Avoid offensive or inappropriate language.
"""
query_wrapper_prompt = PromptTemplate(
    "[INST]<>\n" + SYSTEM_PROMPT + "<>\n\n{query_str}[/INST] "
)
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=MISTRAL_7B,
    model_name=MISTRAL_7B,
    device_map="auto",
    model_kwargs={},
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

# Load the CSV file with caching
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("GPT-2 Text Generation with Streamlit")

    # Load the CSV file
    csv_file_path = "https://raw.githubusercontent.com/neilh44/AMZ/main/A1.csv"
    df = load_csv(csv_file_path)

    # Display the CSV file
    st.subheader("CSV File Content")
    st.write(df)

    # Text input area for user prompt
    prompt = st.text_area("Enter your prompt here:", height=100)

    # Button to generate text
    if st.button("Generate Text"):
        # Generate text using GPT-2 model
        input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        output_ids = model.generate(input_ids, max_length=200, num_return_sequences=1, early_stopping=True, attention_mask=input_ids.ne(tokenizer.pad_token_id))
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Generate text using Llama Index
        generated_text_llm = llm.generate(prompt)

        # Display generated text from GPT-2 and Llama Index
        st.subheader("Generated Text (GPT-2):")
        st.write(generated_text)
        st.subheader("Generated Text (Llama Index):")
        st.write(generated_text_llm)

# Run the Streamlit app
if __name__ == "__main__":
    main()
