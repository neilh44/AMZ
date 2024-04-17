import streamlit as st
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the BART model for summarization
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load the fine-tuned BART model for query response generation (if available)
# query_model = BartForConditionalGeneration.from_pretrained("your_fine_tuned_model_path")
# query_tokenizer = BartTokenizer.from_pretrained("your_fine_tuned_model_path")

# Load the CSV file with caching
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Function to summarize the CSV data using BART
def summarize_data(data):
    inputs = tokenizer.batch_encode_plus(data, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
    summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return summaries

# Function to generate query responses using BART (if available)
# def generate_query_response(prompt):
#     inputs = query_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
#     response_ids = query_model.generate(inputs, max_length=150, num_return_sequences=1, early_stopping=True)
#     responses = [query_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in response_ids]
#     return responses

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("BART Text Summarization and Query Response Generation")

    # Load the CSV file
    csv_file_path = "https://raw.githubusercontent.com/neilh44/AMZ/main/A1C2.csv"
    df = load_csv(csv_file_path)

    # Display the CSV file
    st.subheader("CSV File Content")
    st.write(df)

    # Summarize the CSV data
    st.subheader("Summarized Data")
    summaries = summarize_data(df)
    for summary in summaries:
        st.write(summary)

    # Text input area for user prompt
    # prompt = st.text_area("Enter your prompt here:", height=100)

    # Button to generate query response
    # if st.button("Generate Query Response"):
    #     responses = generate_query_response(prompt)
    #     for response in responses:
    #         st.write(response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
