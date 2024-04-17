import streamlit as st
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load the CSV file with caching
@st.cache
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
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        # Generate text
        outputs = model.generate(**inputs)

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display generated text
        st.subheader("Generated Text:")
        st.write(generated_text)

if __name__ == "__main__":
    main()
