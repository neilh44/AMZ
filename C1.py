# Import necessary libraries
import streamlit as st
import pandas as pd
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load the CSV file with caching
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Load the language model using llama index
@st.cache_resource
def load_llm():
    llm = HuggingFaceLLM(
        model_name="gpt2",  # or any other model from Hugging Face Transformers
        device_map="auto",
        model_kwargs={},
    )
    Settings.llm = llm

# Function to generate text using the language model
def generate_text(prompt):
    return Settings.llm.generate(prompt)

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("Language Model Text Generation with Streamlit")

    # Load the CSV file
    csv_file_path = "https://raw.githubusercontent.com/neilh44/AMZ/main/A1C2.csv"
    df = load_csv(csv_file_path)

    # Display the CSV file (optional)
    st.subheader("CSV File Content")
    st.write(df)

    # Load the language model
    load_llm()

    # Text input area for user prompt
    prompt = st.text_area("Enter your prompt here:", height=100)

    # Button to generate text
    if st.button("Generate Text"):
        # Generate text
        generated_text = generate_text(prompt)

        # Display generated text
        st.subheader("Generated Text:")
        st.write(generated_text)

# Run the Streamlit app
if __name__ == "__main__":
    main()
