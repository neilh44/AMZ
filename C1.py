import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index import Llama_Index

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the CSV file
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Set up the Llama index with the CSV file
@st.cache(allow_output_mutation=True)
def setup_llama_index(file_path):
    index = LlamaIndex()
    index.index_csv(file_path)
    return index

# Define the Streamlit app
def main():
    st.title("Text Generation with Mistral 7B")

    # File upload for CSV file
    file_upload = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if file_upload is not None:
        # Load the CSV file
        df = load_data(file_upload)
        st.write("CSV File Preview:", df)

        # Set up the Llama index
        index = setup_llama_index(file_upload)

        # Text input area for user query
        query = st.text_input("Enter your query here:")

        # Button to execute query
        if st.button("Execute Query"):
            # Execute query on the index
            results = index.query(query)
            st.write("Query Results:", results)

            # Use query results as input prompts for text generation
            for i, result in enumerate(results):
                # Generate text
                input_ids = tokenizer.encode(result, return_tensors="pt")
                output = model.generate(input_ids, max_length=100, num_return_sequences=3, temperature=0.9)

                # Decode and display generated text
                for j, sample_output in enumerate(output):
                    st.subheader(f"Generated Text {i + 1}-{j + 1}:")
                    st.write(tokenizer.decode(sample_output, skip_special_tokens=True))

if __name__ == "__main__":
    main()
