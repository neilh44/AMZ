import streamlit as st
import pandas as pd
from langchain.llm import LangChainLLM
from langchain.core import PromptTemplate, Settings

# Load the CSV file
@st.cache
def load_csv(file_path):
    return pd.read_csv(file_path)

# Load the BART model
def load_bart_model():
    # Define model name
    BART_MODEL = "facebook/bart-large-cnn"

    # Initialize the LangChainLLM instance with BART model
    llm = LangChainLLM(
        model_name=BART_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Set llm in Settings
    Settings.llm = llm

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("BART Text Generation with Streamlit")

    # Load the CSV file
    csv_file_path = "https://raw.githubusercontent.com/neilh44/AMZ/main/A1C2.csv"
    df = load_csv(csv_file_path)

    # Display the CSV file
    st.subheader("CSV File Content")
    st.write(df)

    # Load the BART model
    load_bart_model()

    # Text input area for user prompt
    prompt = st.text_area("Enter your prompt here:", height=100)

    # Button to generate text
    if st.button("Generate Text"):
        # Generate text
        generated_text = generate_text(prompt)

        # Display generated text
        st.subheader("Generated Text:")
        st.write(generated_text)

# Function to generate text using BART model
def generate_text(prompt):
    return Settings.llm.generate(prompt)

if __name__ == "__main__":
    main()
