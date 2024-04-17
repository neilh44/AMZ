import streamlit as st
import pandas as pd
import logging
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load the CSV file with custom parsing options
@st.cache_data
def load_csv(file_path):
    # Define custom parsing options
    custom_options = {
        'encoding': 'utf-8',  # Specify the encoding of the CSV file
        'sep': ',',           # Specify the delimiter used in the CSV file (comma-separated)
        'quotechar': '"',     # Specify the quote character used to enclose fields
        'skiprows': 11        # Specify the number of initial rows to skip (excluding header)
    }
    return pd.read_csv(file_path, **custom_options)

# Load the Mistral 7B model with caching as a resource
@st.cache_resource
def load_mistral_model():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Define model name
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.2"

    # Define system prompt
    SYSTEM_PROMPT = """You are an AI assistant that analyzes data from the provided CSV file. Here are some rules you always follow:
    - Generate human-readable output.
    - Generate only the requested output, avoiding unnecessary information.
    - Avoid offensive or inappropriate language.
    """

    # Create query wrapper prompt
    query_wrapper_prompt = PromptTemplate(
        "[INST]<>\n" + SYSTEM_PROMPT + "<>\n\n{query_str}[/INST] "
    )

    # Initialize the HuggingFaceLLM instance with Mistral 7B model
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

    # Define the embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Set llm and embed_model in Settings
    Settings.llm = llm
    Settings.embed_model = embed_model

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("Mistral 7B Text Generation with Streamlit")

    # Load the CSV file
    csv_file_path = "https://raw.githubusercontent.com/neilh44/AMZ/main/A1.csv"
    df = load_csv(csv_file_path)

    # Display the CSV file
    st.subheader("CSV File Content")
    st.write(df)

    # Load the Mistral 7B model
    load_mistral_model()

    # Text input area for user prompt
    prompt = st.text_area("Enter your prompt here:", height=100)

    # Button to generate text
    if st.button("Generate Text"):
        # Generate text
        generated_text = generate_text(prompt)

        # Display generated text
        st.subheader("Generated Text:")
        st.write(generated_text)

# Function to generate text using Mistral 7B model
def generate_text(prompt):
    return Settings.llm.generate(prompt)

if __name__ == "__main__":
    main()
