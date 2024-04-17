import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index import LlamaIndex


# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the Streamlit app
def main():
    st.title("Text Generation with Mistral 7B")

    # Read the CSV file from the GitHub URL
    csv_url = "https://raw.githubusercontent.com/neilh44/AMZ/main/tt4.csv"
    df = pd.read_csv(csv_url)

    # Text input area for user prompt
    prompt = st.text_area("Enter your prompt here:", height=100)

    # Button to generate text
    if st.button("Generate Text"):
        # Combine prompt with CSV data
        data = "\n".join(df.to_string(index=False).splitlines())

        # Encode input prompt
        input_ids = tokenizer.encode(prompt + "\n" + data, return_tensors="pt")

        # Generate text
        output = model.generate(input_ids, max_length=100, num_return_sequences=3, temperature=0.9)

        # Decode and display generated text
        for i, sample_output in enumerate(output):
            st.subheader(f"Generated Text {i + 1}:")
            st.write(tokenizer.decode(sample_output, skip_special_tokens=True))

if __name__ == "__main__":
    main()
