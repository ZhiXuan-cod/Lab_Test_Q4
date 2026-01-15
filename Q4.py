import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt')

st.title("Text Chunking with NLTK")
st.markdown("### Semantic Sentence Segmentation")

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])

if uploaded_file is not None:
    # Step 1: Read PDF
    reader = PdfReader(uploaded_file)
    
    # Step 2: Extract text
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    
    st.success(f"PDF loaded successfully! Pages: {len(reader.pages)}")
    
    # Display extracted text stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Characters", f"{len(full_text):,}")
    with col2:
        st.metric("Pages", len(reader.pages))
    with col3:
        st.metric("Lines", full_text.count('\n'))
    
    
    # Split into sentences (basic)
    sentences = full_text.split('. ')
    
    st.markdown(f"**Total sentences (basic split)**: {len(sentences)}")
    
    # Display sample (indices 58-68)
    st.markdown("**Sample Text (Indices 58-68):**")
    sample_text = ""
    for i in range(58, min(69, len(sentences))):
        if i < len(sentences):
            sample_text += f"{i}: {sentences[i][:100]}...\n\n"
    
    st.text_area("Text Sample", sample_text, height=200)
    
    
    if st.button("Perform Sentence Tokenization"):
        with st.spinner("Tokenizing sentences..."):
            # Use NLTK's sentence tokenizer
            nltk_sentences = sent_tokenize(full_text)
            
            st.success(f"NLTK identified {len(nltk_sentences)} semantic sentences")
            
            # Display tokenized sentences
            st.markdown("**First 10 Tokenized Sentences:**")
            
            tokenized_data = []
            for i, sent in enumerate(nltk_sentences[:10]):
                tokenized_data.append({
                    'Sentence #': i+1,
                    'Text Preview': sent[:80] + "..." if len(sent) > 80 else sent,
                    'Length': len(sent),
                    'Word Count': len(sent.split())
                })
            
            st.table(tokenized_data)
            
            # Comparison between methods
            st.subheader("Method Comparison")
            
            comparison_data = {
                'Method': ['Basic Split (.)', 'NLTK Tokenizer'],
                'Sentences Found': [len(sentences), len(nltk_sentences)],
                'Avg Sentence Length': [
                    sum(len(s) for s in sentences)/len(sentences) if sentences else 0,
                    sum(len(s) for s in nltk_sentences)/len(nltk_sentences) if nltk_sentences else 0
                ],
                'Semantic Accuracy': ['Low', 'High']
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.table(df_comparison)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Sentence length distribution
            lengths = [len(sent.split()) for sent in nltk_sentences]
            ax1.hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Words per Sentence')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Sentence Length Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Method comparison bar chart
            methods = comparison_data['Method']
            counts = comparison_data['Sentences Found']
            bars = ax2.bar(methods, counts, color=['lightcoral', 'lightgreen'])
            ax2.set_ylabel('Number of Sentences')
            ax2.set_title('Sentence Count by Method')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 5,
                        f'{count}', ha='center')
            
            st.pyplot(fig)
            
            # Analysis
            st.subheader("Technical Analysis")
            st.markdown("""
            **NLTK Sentence Tokenizer Advantages:**
            1. **Semantic Awareness**: Handles abbreviations (Dr., etc.)
            2. **Punctuation Handling**: Correctly processes ? ! and .
            3. **Context Understanding**: Better with decimal points, URLs, emails
            
            **Key Observations:**
            - NLTK typically finds more sentences than simple split
            - Better handling of complex punctuation
            - More accurate for technical/scientific text
            - Handles edge cases (Mr., Dr., Inc., etc.)
            
            **Use Cases:**
            - Document summarization
            - Text analytics
            - Natural language understanding
            - Search engine optimization
            """)
            
            # Download option
            st.download_button(
                label="Download Tokenized Sentences",
                data="\n\n".join([f"Sentence {i+1}: {sent}" 
                                 for i, sent in enumerate(nltk_sentences)]),
                file_name="tokenized_sentences.txt",
                mime="text/plain"
            )

# Instructions
with st.expander("Implementation Steps"):
    st.markdown("""
    **Step-by-Step Implementation:**
    
    1. **PDF Reading**: 
       ```python
       from PyPDF2 import PdfReader
       reader = PdfReader(pdf_file)
       ```
    
    2. **Text Extraction**:
       ```python
       text = ""
       for page in reader.pages:
           text += page.extract_text()
       ```
    
    3. **NLTK Setup**:
       ```python
       import nltk
       nltk.download('punkt')
       from nltk.tokenize import sent_tokenize
       ```
    
    4. **Sentence Tokenization**:
       ```python
       sentences = sent_tokenize(text)
       ```
    
    **Key Features:**
    - Handles multiple PDF pages
    - Semantic sentence boundary detection
    - Statistical comparison of methods
    - Visualization of results
    - Export capability
    """)

