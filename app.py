import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="News NER Extractor", layout="wide")
st.title("📰 News Information Extraction")

@st.cache_resource
def load_model():
    # This path must match your trainer.save_model path
    model_path = "./bert-finetuned-ner" 
    return pipeline("ner", model=model_path, tokenizer=model_path, aggregation_strategy="simple")

try:
    ner_pipe = load_model()
    text_input = st.text_area("Enter News Article Text:", height=150)

    if st.button("Extract Entities"):
        if text_input:
            results = ner_pipe(text_input)
            if results:
                st.subheader("Extracted Entities:")
                df = pd.DataFrame(results)[['word', 'entity_group', 'score']]
                df.columns = ['Entity', 'Type', 'Confidence']
                st.table(df)
            else:
                st.info("No entities detected.")
except Exception as e:
    st.error(f"Error: {e}")
