import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Mistral 7B model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate text based on user query
def generate_text(query, csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert the query and file content into a single prompt
    prompt = query + "\n" + df.to_csv(index=False)

    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.9)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Define the Streamlit app
def main():
    st.title("CSV Analyzer with Mistral 7B")

    # File upload for CSV file
    st.sidebar.title("Upload CSV File")
    csv_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    # Text input area for user query
    query = st.text_area("Enter your query here:", height=100)

    # Button to generate response
    if st.button("Generate Response"):
        if csv_file is not None:
            # Generate response based on user query and uploaded CSV file
            response = generate_text(query, csv_file)
            st.subheader("Generated Response:")
            st.write(response)
        else:
            st.error("Please upload a CSV file.")

if __name__ == "__main__":
    main()
