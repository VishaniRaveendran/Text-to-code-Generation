import streamlit as st
import random
from datasets import load_dataset
from transformers import *
import tensorflow as tf

# @st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    save_dir = "saved_model/"
    
    # load saved finetuned model
    model = TFT5ForConditionalGeneration.from_pretrained(save_dir)
    # load saved tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(save_dir)
    return model, tokenizer

def run_predict( text, model, tokenizer):
    # encode texts by prepending the task for input sequence and appending the test sequence
    # DATA
    max_input_length = 48
    max_target_length = 128
    prefix = "Generate Python: "  
    query = prefix + text 
    encoded_text = tokenizer(query, return_tensors='tf', padding='max_length', truncation=True, max_length=max_input_length)
    
    # inference
    generated_code = model.generate(
        encoded_text["input_ids"], attention_mask=encoded_text["attention_mask"], 
        max_length=max_target_length, top_p=0.95, top_k=50, repetition_penalty=2.00, num_return_sequences=1
    )
    
    # decode generated tokens
    decoded_code = tokenizer.decode(generated_code.numpy()[0], skip_special_tokens=True)
    return decoded_code

def predict_from_dataset(model, tokenizer):
    # load using hf datasets
    dataset = load_dataset('json', data_files='mbpp.jsonl') 

    # train test split
    dataset = dataset['train'].train_test_split(0.1, shuffle=False) 
    test_dataset = dataset['test']
    
    # randomly select an index from the validation dataset
    index = random.randint(0, len(test_dataset))
    text = test_dataset[index]['text']
    code = test_dataset[index]['code']
    
    # run-predict on text
    decoded_code = run_predict(text, model, tokenizer)
    
    st.text("#" * 25)
    st.text("QUERY: " + text)
    st.text('#' * 25)
    st.text("ORIGINAL:")
    st.text("\n" + code)
    st.text('#' * 25)
    st.text("GENERATED:")
    st.text("\n" + decoded_code)

def predict_from_text( text, model, tokenizer):
    # run-predict on text
    decoded_code = run_predict( text, model, tokenizer)
    st.text("#" * 25)
    st.text("QUERY: " + text)
    st.text('#' * 25)
    st.text("GENERATED:")
    st.text("\n" + decoded_code)

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Streamlit app
st.title("Code Generation")
option = st.sidebar.selectbox("Select Mode", ["Dataset", "Text"])

if option == "Dataset":
    predict_from_dataset(model, tokenizer)
else:
    text_input = st.text_area("Enter Text", "")
    if st.button("Generate"):
        predict_from_text( text_input, model, tokenizer)
