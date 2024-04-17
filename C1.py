# Import necessary libraries
import streamlit as st
import pandas as pd
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, Settings

# Load the language model using llama index
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
    csv_file_path = "your_csv_file_path"

    # Display the CSV file (optional)
    df = pd.read_csv(csv_file_path)
    st.subheader("CSV File Content")
    st.write(df)

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
