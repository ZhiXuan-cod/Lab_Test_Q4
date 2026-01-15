import streamlit as st
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

st.title("Text Chunking App")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in reader.pages)

    sentences = sent_tokenize(text)
    sample = sentences[58:68]

    st.subheader("Extracted Sentences (58–68)")
    for s in sample:
        st.write(s)

    st.subheader("Semantic Chunks")
    chunks = sent_tokenize(" ".join(sample))
    for c in chunks:
        st.write("-", c)
