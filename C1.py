import streamlit as st
import pandas as pd
import logging
from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load the CSV file with caching
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("BART Text Generation with Streamlit")

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
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)

        # Generate text
        output_ids = model.generate(input_ids, max_length=200, num_return_sequences=1, early_stopping=True)

        # Decode generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Display generated text
        st.subheader("Generated Text:")
        st.write(generated_text)

# Function to generate text using BART model
if __name__ == "__main__":
    main()
