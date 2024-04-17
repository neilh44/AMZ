import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the Streamlit app
def main():
    st.title("Text Generation with Mistral 7B")

    # File upload for CSV file
    st.sidebar.title("Upload CSV File")
    file_upload = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    # Text input area for user prompt
    prompt = st.text_area("Enter your prompt here:", height=100)

    # Button to generate text
    if st.button("Generate Text"):

        if file_upload is not None:
            # Read the file content
            file_content = file_upload.read()
            prompt += "\n" + file_content

        # Encode input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate text
        output = model.generate(input_ids, max_length=100, num_return_sequences=3, temperature=0.9)

        # Decode and display generated text
        for i, sample_output in enumerate(output):
            st.subheader(f"Generated Text {i + 1}:")
            st.write(tokenizer.decode(sample_output, skip_special_tokens=True))

if __name__ == "__main__":
    main()
