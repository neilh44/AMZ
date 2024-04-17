import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token_id to eos_token_id for open-end generation
model.config.pad_token_id = model.config.eos_token_id

# Load the CSV file with caching
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("GPT-2 Text Generation with Streamlit")

    # Load the CSV file
    csv_file_path = "https://raw.githubusercontent.com/neilh44/AMZ/main/tt4.csv"
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
        output_ids = model.generate(input_ids, max_length=200, num_return_sequences=1, early_stopping=True, attention_mask=input_ids.ne(tokenizer.pad_token_id))

        # Decode generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Display generated text
        st.subheader("Generated Text:")
        st.write(generated_text)

# Function to generate text using GPT-2 model
if __name__ == "__main__":
    main()
