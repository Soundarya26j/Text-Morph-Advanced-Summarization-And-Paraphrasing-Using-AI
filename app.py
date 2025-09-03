import os
import io
import datetime
import bcrypt
import jwt
import streamlit as st
import sqlite3
import pandas as pd

from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, select
from sqlalchemy.orm import declarative_base, sessionmaker

import textstat
import nltk

# Download NLTK data with error handling
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("tokenizers/punkt", quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")
    print("Please run: python -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"punkt_tab\")'")

from nltk.tokenize import sent_tokenize

# Fallback sentence tokenization function
def safe_sent_tokenize(text: str) -> list:
    """Safe sentence tokenization with fallback."""
    try:
        return sent_tokenize(text)
    except Exception as e:
        print(f"Warning: NLTK sentence tokenization failed: {e}")
        # Simple fallback: split on common sentence endings
        import re
        sentences = re.split(r'[.!?]+', text)
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [text]

try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader

# Summarization imports
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    from rouge_score import rouge_scorer
    import re
    SUMMARIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Summarization dependencies not available: {e}")
    print("Please install: pip install rouge-score accelerate")
    SUMMARIZATION_AVAILABLE = False

# Paraphrasing imports
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    PARAPHRASING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Paraphrasing dependencies not available: {e}")
    print("Please install: pip install sentence-transformers")
    PARAPHRASING_AVAILABLE = False

st.set_page_config(page_title="File Preview App", layout="wide")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #b57edc, #006994);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #b57edc, #006994);
    }
    .login-card, .signup-card {
        max-width: 400px;
        margin: 90px auto;
        padding: 24px;
        border-radius: 16px;
        background: rgba(0,0,0,0.55);
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
        text-align: center;
    }
    .login-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 18px;
        color: #FFD700;
    }

    /* Inputs / selects styling */
    .stTextInput > div > div > input,
    .stPasswordInput > div > div > input,
    .stSelectbox div[role="combobox"] select {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        font-size: 15px !important;
        height: 35px !important;
        padding: 6px 10px !important;
    }

    /* Buttons full width inside cards */
    .stButton button {
        background: linear-gradient(90deg, #2563EB, #4F46E5);
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: 600;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #1D4ED8, #4338CA);
    }

    /* Keep checkboxes left-aligned inside card */
    .stCheckbox {
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./auth.db")

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(120), nullable=True)
    password_hash = Column(String(255), nullable=False)
    language = Column(String(20), default="English")
    age_group = Column(String(20), default="18-25")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

@st.cache_resource
def get_engine():
    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    return create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)

@st.cache_resource
def get_sessionmaker():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

SessionLocal = get_sessionmaker()

# Summarization models
@st.cache_resource
def load_summarization_models():
    """Load and cache summarization models."""
    if not SUMMARIZATION_AVAILABLE:
        st.error("Summarization features are not available. Please install required dependencies.")
        return {}
    
    models = {}
    
    try:
        # PEGASUS model
        models['pegasus'] = pipeline(
            "summarization",
            model="google/pegasus-xsum",
            tokenizer="google/pegasus-xsum",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.warning(f"Could not load PEGASUS model: {e}")
    
    try:
        # FLAN-T5 model
        models['flan-t5'] = pipeline(
            "summarization",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.warning(f"Could not load FLAN-T5 model: {e}")
    
    try:
        # BART model
        models['bart'] = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.warning(f"Could not load BART model: {e}")
    
    return models

# Paraphrasing models
@st.cache_resource
def load_paraphrasing_models():
    """Load and cache paraphrasing models."""
    if not PARAPHRASING_AVAILABLE:
        st.error("Paraphrasing features are not available. Please install required dependencies.")
        return {}
    
    models = {}
    
    try:
        # FLAN-T5 for paraphrasing
        models['flan-t5-paraphrase'] = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.warning(f"Could not load FLAN-T5 paraphrasing model: {e}")
    
    try:
        # BART for paraphrasing
        models['bart-paraphrase'] = pipeline(
            "text2text-generation",
            model="facebook/bart-large",
            tokenizer="facebook/bart-large",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.warning(f"Could not load BART paraphrasing model: {e}")
    
    return models

def get_complexity_prompt(complexity_level: str) -> str:
    """Get prompt for different complexity levels."""
    prompts = {
        "simplified": "Rewrite this text in simpler language that is easier to understand: ",
        "standard": "Paraphrase this text while maintaining the same meaning: ",
        "enhanced": "Rewrite this text with more sophisticated vocabulary and complex sentence structures: ",
        "academic": "Rewrite this text in formal academic style with advanced terminology: "
    }
    return prompts.get(complexity_level, prompts["standard"])

def paraphrase_text(text: str, model_name: str, complexity_level: str = "standard") -> str:
    """Paraphrase text using specified model and complexity level."""
    try:
        models = load_paraphrasing_models()
        if model_name not in models:
            return f"Model {model_name} not available"
        
        model = models[model_name]
        prompt = get_complexity_prompt(complexity_level)
        
        # Prepare input with prompt
        input_text = prompt + text
        
        # Generate paraphrase
        result = model(input_text, 
                     max_length=len(text.split()) * 2,  # Allow some flexibility in length
                     min_length=len(text.split()) // 2,  # Ensure minimum length
                     do_sample=True,
                     temperature=0.7,
                     num_return_sequences=1)
        
        paraphrased = result[0]['generated_text']
        
        # Clean up the output (remove prompt if it appears)
        if paraphrased.startswith(prompt):
            paraphrased = paraphrased[len(prompt):]
        
        return paraphrased.strip()
        
    except Exception as e:
        return f"Error generating paraphrase: {str(e)}"

def calculate_similarity_score(original: str, paraphrased: str) -> float:
    """Calculate semantic similarity between original and paraphrased text."""
    try:
        if not PARAPHRASING_AVAILABLE:
            return 0.0
        
        # Load sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode both texts
        embeddings = model.encode([original, paraphrased])
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def split_into_sentences_and_paragraphs(text: str) -> tuple:
    """Split text into sentences and paragraphs."""
    # Split into paragraphs (double newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Split into sentences using safe tokenization
    sentences = safe_sent_tokenize(text)
    
    return sentences, paragraphs

def paraphrase_text_ui(text: str):
    """UI for text paraphrasing with multiple models and complexity levels."""
    st.subheader("🔄 Text Paraphrasing")
    
    if not PARAPHRASING_AVAILABLE:
        st.error("""
        **Paraphrasing features are not available.**
        
        Please install the required dependencies by running:
        ```
        pip install sentence-transformers
        ```
        
        Then restart the application.
        """)
        return
    
    if not text or len(text.strip()) < 10:
        st.warning("Text is too short for meaningful paraphrasing. Please upload a longer document.")
        return
    
    # Model and complexity selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_options = ["flan-t5-paraphrase", "bart-paraphrase"]
        selected_model = st.selectbox(
            "Choose Paraphrasing Model",
            model_options,
            help="FLAN-T5: Good for general paraphrasing. BART: Good for creative rewording."
        )
    
    with col2:
        complexity_options = ["simplified", "standard", "enhanced", "academic"]
        selected_complexity = st.selectbox(
            "Complexity Level",
            complexity_options,
            help="Simplified: Easier language. Standard: Same level. Enhanced: More sophisticated. Academic: Formal style."
        )
    
    # Paraphrasing level selection
    paraphrase_level = st.radio(
        "Paraphrasing Level",
        ["Sentence Level", "Paragraph Level", "Full Text"],
        horizontal=True,
        help="Sentence: Paraphrase each sentence individually. Paragraph: Paraphrase each paragraph. Full Text: Paraphrase entire text."
    )
    
    # Generate paraphrase button
    if st.button("Generate Paraphrase", type="primary"):
        with st.spinner("Generating paraphrase..."):
            
            if paraphrase_level == "Sentence Level":
                # Paraphrase each sentence
                sentences, _ = split_into_sentences_and_paragraphs(text)
                paraphrased_sentences = []
                
                for sentence in sentences:
                    if len(sentence.strip()) > 5:  # Only paraphrase substantial sentences
                        paraphrased = paraphrase_text(sentence, selected_model, selected_complexity)
                        paraphrased_sentences.append(paraphrased)
                    else:
                        paraphrased_sentences.append(sentence)
                
                paraphrased_text = " ".join(paraphrased_sentences)
                
            elif paraphrase_level == "Paragraph Level":
                # Paraphrase each paragraph
                _, paragraphs = split_into_sentences_and_paragraphs(text)
                paraphrased_paragraphs = []
                
                for paragraph in paragraphs:
                    if len(paragraph.strip()) > 20:  # Only paraphrase substantial paragraphs
                        paraphrased = paraphrase_text(paragraph, selected_model, selected_complexity)
                        paraphrased_paragraphs.append(paraphrased)
                    else:
                        paraphrased_paragraphs.append(paragraph)
                
                paraphrased_text = "\n\n".join(paraphrased_paragraphs)
                
            else:  # Full Text
                paraphrased_text = paraphrase_text(text, selected_model, selected_complexity)
            
            if paraphrased_text and not paraphrased_text.startswith("Error"):
                st.success("Paraphrase generated successfully!")
                
                # Calculate similarity score
                similarity = calculate_similarity_score(text, paraphrased_text)
                
                # Display comparison
                st.subheader("📊 Original vs Paraphrased Text")
                
                # Side-by-side comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Text**")
                    st.text_area("Original", text, height=300, disabled=True)
                
                with col2:
                    st.markdown("**Paraphrased Text**")
                    st.text_area("Paraphrased", paraphrased_text, height=300, disabled=True)
                
                # Similarity score
                st.subheader("📈 Paraphrase Quality Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Semantic Similarity", f"{similarity:.3f}")
                    if similarity >= 0.8:
                        st.success("High similarity")
                    elif similarity >= 0.6:
                        st.warning("Moderate similarity")
                    else:
                        st.error("Low similarity")
                
                with col2:
                    st.metric("Original Words", len(text.split()))
                
                with col3:
                    st.metric("Paraphrased Words", len(paraphrased_text.split()))
                
                with col4:
                    word_change = abs(len(paraphrased_text.split()) - len(text.split()))
                    st.metric("Word Difference", word_change)
                
                # Detailed analysis
                with st.expander("🔍 Detailed Analysis"):
                    st.markdown("""
                    **Semantic Similarity Score:**
                    - **0.8-1.0:** Excellent paraphrase (maintains meaning well)
                    - **0.6-0.8:** Good paraphrase (some meaning preserved)
                    - **0.4-0.6:** Fair paraphrase (partial meaning preserved)
                    - **0.0-0.4:** Poor paraphrase (meaning may be lost)
                    
                    **Complexity Levels:**
                    - **Simplified:** Uses simpler vocabulary and shorter sentences
                    - **Standard:** Maintains original complexity level
                    - **Enhanced:** Uses more sophisticated vocabulary and complex structures
                    - **Academic:** Formal academic style with advanced terminology
                    """)
                
                # Download options
                st.subheader("💾 Download Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create download for paraphrased text
                    paraphrased_bytes = paraphrased_text.encode()
                    st.download_button(
                        label="Download Paraphrased Text",
                        data=paraphrased_bytes,
                        file_name="paraphrased_text.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Create comparison file
                    comparison_text = f"ORIGINAL TEXT:\n{text}\n\nPARAPHRASED TEXT:\n{paraphrased_text}\n\nSIMILARITY SCORE: {similarity:.3f}"
                    comparison_bytes = comparison_text.encode()
                    st.download_button(
                        label="Download Comparison",
                        data=comparison_bytes,
                        file_name="text_comparison.txt",
                        mime="text/plain"
                    )
                
            else:
                st.error(f"Failed to generate paraphrase: {paraphrased_text}")

def get_summary_length_config(length_type: str) -> dict:
    """Get configuration for different summary lengths."""
    configs = {
        "short": {
            "max_length": 50,
            "min_length": 20,
            "length_penalty": 0.8
        },
        "medium": {
            "max_length": 150,
            "min_length": 50,
            "length_penalty": 1.0
        },
        "long": {
            "max_length": 300,
            "min_length": 100,
            "length_penalty": 1.2
        }
    }
    return configs.get(length_type, configs["medium"])

def generate_summary(text: str, model_name: str, length_type: str = "medium") -> str:
    """Generate summary using specified model and length."""
    try:
        models = load_summarization_models()
        if model_name not in models:
            return f"Model {model_name} not available"
        
        model = models[model_name]
        config = get_summary_length_config(length_type)
        
        # Split text into chunks if too long
        max_input_length = 1024
        if len(text) > max_input_length:
            # Split into sentences and create chunks
            sentences = safe_sent_tokenize(text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < max_input_length:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Generate summary for each chunk
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                    summary = model(chunk, **config)[0]['summary_text']
                    summaries.append(summary)
            
            # Combine summaries
            if summaries:
                combined_summary = " ".join(summaries)
                # Generate final summary if combined summary is still long
                if len(combined_summary) > max_input_length:
                    final_summary = model(combined_summary, **config)[0]['summary_text']
                    return final_summary
                return combined_summary
            else:
                return "Text too short to summarize"
        else:
            # Direct summarization
            summary = model(text, **config)[0]['summary_text']
            return summary
            
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def calculate_rouge_scores(original_text: str, summary: str) -> dict:
    """Calculate ROUGE scores for summary evaluation."""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(original_text, summary)
        
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'fmeasure': scores['rougeL'].fmeasure
            }
        }
    except Exception as e:
        return {
            'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
            'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
            'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
        }

def summarize_text_ui(text: str):
    """UI for text summarization with multiple models and lengths."""
    st.subheader("📝 Text Summarization")
    
    if not SUMMARIZATION_AVAILABLE:
        st.error("""
        **Summarization features are not available.**
        
        Please install the required dependencies by running:
        ```
        pip install rouge-score accelerate
        ```
        
        Then restart the application.
        """)
        return
    
    if not text or len(text.strip()) < 100:
        st.warning("Text is too short for meaningful summarization. Please upload a longer document.")
        return
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_options = ["pegasus", "flan-t5", "bart"]
        selected_model = st.selectbox(
            "Choose Summarization Model",
            model_options,
            help="PEGASUS: Good for news articles. FLAN-T5: Versatile and accurate. BART: Good for general text."
        )
    
    with col2:
        length_options = ["short", "medium", "long"]
        selected_length = st.selectbox(
            "Summary Length",
            length_options,
            help="Short: ~50 words, Medium: ~150 words, Long: ~300 words"
        )
    
    # Generate summary button
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            summary = generate_summary(text, selected_model, selected_length)
            
            if summary and not summary.startswith("Error"):
                st.success("Summary generated successfully!")
                
                # Display summary
                st.subheader("📋 Generated Summary")
                st.text_area("Summary", summary, height=200, disabled=True)
                
                # Calculate and display ROUGE scores
                st.subheader("📊 Summary Evaluation (ROUGE Scores)")
                rouge_scores = calculate_rouge_scores(text, summary)
                
                # Display ROUGE scores in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("ROUGE-1 F1", f"{rouge_scores['rouge1']['fmeasure']:.3f}")
                    st.caption(f"Precision: {rouge_scores['rouge1']['precision']:.3f}")
                    st.caption(f"Recall: {rouge_scores['rouge1']['recall']:.3f}")
                
                with col2:
                    st.metric("ROUGE-2 F1", f"{rouge_scores['rouge2']['fmeasure']:.3f}")
                    st.caption(f"Precision: {rouge_scores['rouge2']['precision']:.3f}")
                    st.caption(f"Recall: {rouge_scores['rouge2']['recall']:.3f}")
                
                with col3:
                    st.metric("ROUGE-L F1", f"{rouge_scores['rougeL']['fmeasure']:.3f}")
                    st.caption(f"Precision: {rouge_scores['rougeL']['precision']:.3f}")
                    st.caption(f"Recall: {rouge_scores['rougeL']['recall']:.3f}")
                
                # ROUGE score explanation
                with st.expander("ℹ️ Understanding ROUGE Scores"):
                    st.markdown("""
                    **ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Scores:**
                    
                    - **ROUGE-1:** Measures word overlap between summary and original text
                    - **ROUGE-2:** Measures bigram (2-word) overlap
                    - **ROUGE-L:** Measures longest common subsequence
                    
                    **Score Interpretation:**
                    - **0.0-0.2:** Poor summary quality
                    - **0.2-0.4:** Fair summary quality  
                    - **0.4-0.6:** Good summary quality
                    - **0.6+:** Excellent summary quality
                    
                    **F1 Score:** Balanced measure of precision and recall
                    """)
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Original Words", len(text.split()))
                with col2:
                    st.metric("Summary Words", len(summary.split()))
                with col3:
                    compression_ratio = len(summary.split()) / len(text.split()) * 100
                    st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
                with col4:
                    st.metric("Model Used", selected_model.upper())
                
            else:
                st.error(f"Failed to generate summary: {summary}")

def ensure_uploads_table():
    conn = sqlite3.connect("file_uploads.db")
    cur = conn.cursor()
    
    # Create table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            filename TEXT,
            filedata BLOB,
            readability_level TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    
    # Check if required columns exist and add them if missing
    try:
        cur.execute("PRAGMA table_info(uploads)")
        columns = [column[1] for column in cur.fetchall()]
        
        if 'filedata' not in columns:
            cur.execute("ALTER TABLE uploads ADD COLUMN filedata BLOB")
            print("Added missing filedata column to uploads table")
            
        if 'readability_level' not in columns:
            cur.execute("ALTER TABLE uploads ADD COLUMN readability_level TEXT")
            print("Added missing readability_level column to uploads table")
            
    except Exception as e:
        print(f"Warning: Could not check/add columns: {e}")
    
    conn.commit()
    conn.close()

def save_upload(username: str, filename: str, data: bytes, readability_level: str = None):
    conn = sqlite3.connect("file_uploads.db")
    cur = conn.cursor()
    
    # Check if filepath column exists
    cur.execute("PRAGMA table_info(uploads)")
    columns = [column[1] for column in cur.fetchall()]
    
    if 'filepath' in columns:
        # Use the existing schema with filepath
        cur.execute("INSERT INTO uploads (username, filename, filepath, filedata, readability_level) VALUES (?, ?, ?, ?, ?)",
                    (username, filename, f"/uploads/{filename}", data, readability_level))
    else:
        # Use the new schema without filepath
        cur.execute("INSERT INTO uploads (username, filename, filedata, readability_level) VALUES (?, ?, ?, ?)",
                    (username, filename, data, readability_level))
    
    conn.commit()
    conn.close()

def list_uploads(username: str):
    conn = sqlite3.connect("file_uploads.db")
    cur = conn.cursor()
    
    # Check if filepath column exists
    cur.execute("PRAGMA table_info(uploads)")
    columns = [column[1] for column in cur.fetchall()]
    
    if 'filepath' in columns:
        # Use the existing schema with filepath
        cur.execute("SELECT id, filename, readability_level, created_at FROM uploads WHERE username=? ORDER BY id DESC", (username,))
    else:
        # Use the new schema without filepath
        cur.execute("SELECT id, filename, readability_level, created_at FROM uploads WHERE username=? ORDER BY id DESC", (username,))
    
    rows = cur.fetchall()
    conn.close()
    return rows

def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def extract_text_from_csv_bytes(b: bytes) -> str:
    try:
        with io.BytesIO(b) as f:
            df = pd.read_csv(f)
        return "\n".join(df.astype(str).fillna("").values.flatten())
    except Exception:
        return ""

def extract_text_from_excel_bytes(b: bytes) -> str:
    try:
        with io.BytesIO(b) as f:
            df = pd.read_excel(f)
        return "\n".join(df.astype(str).fillna("").values.flatten())
    except Exception:
        return ""

def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        with io.BytesIO(b) as f:
            reader = PdfReader(f)
            text = []
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception:
        return ""

def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "txt":
        return extract_text_from_txt_bytes(file_bytes)
    elif ext == "csv":
        return extract_text_from_csv_bytes(file_bytes)
    elif ext in ("xls", "xlsx"):
        return extract_text_from_excel_bytes(file_bytes)
    elif ext == "pdf":
        return extract_text_from_pdf_bytes(file_bytes)
    else:
        return ""

def calculate_readability_scores(text: str) -> dict:
    """
    Calculate overall readability scores for the entire text.
    Returns a dictionary with FK, GF, and SMOG scores.
    """
    try:
        # Calculate scores for the entire text
        fk_score = textstat.flesch_reading_ease(text)
        gf_score = textstat.gunning_fog(text)
        smog_score = textstat.smog_index(text)
        
        return {
            "flesch_kincaid": fk_score,
            "gunning_fog": gf_score,
            "smog_index": smog_score
        }
    except Exception as e:
        print(f"Error calculating readability scores: {e}")
        return {
            "flesch_kincaid": None,
            "gunning_fog": None,
            "smog_index": None
        }
def classify_sentence_level(sentence: str) -> str:
    """
    Compute basic per-sentence readability metrics and classify.
    Uses a heuristic combining Flesch Reading Ease (higher = easier),
    Gunning Fog & SMOG (higher = harder).
    """

    if not sentence or len(sentence.strip()) < 5:
        return "Beginner"

    try:
        
        fk = textstat.flesch_reading_ease(sentence)
        gf = textstat.gunning_fog(sentence)
        sm = textstat.smog_index(sentence)
    except Exception:
        return "Intermediate"

    if fk is None:
        return "Intermediate"

    if fk >= 60 and (gf <= 10 or sm <= 8):
        return "Beginner"
    if 40 <= fk < 60:
        return "Intermediate"
    return "Advanced"

def classify_text(text: str) -> dict:
    """
    Split text into sentences, classify each, and return counts.
    """
    results = {"Beginner": 0, "Intermediate": 0, "Advanced": 0}
    try:
        sentences = safe_sent_tokenize(text)
    except Exception:
        sentences = [text]

    if len(sentences) == 0:
        return results

    for s in sentences:
        lvl = classify_sentence_level(s)
        if lvl in results:
            results[lvl] += 1

    return results

def readability_analysis_ui(text: str):
    st.subheader("📊 Readability Analysis")
    
    # Calculate overall readability scores
    scores = calculate_readability_scores(text)
    
    # Display readability scores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if scores["flesch_kincaid"] is not None:
            st.metric("Flesch-Kincaid", f"{scores['flesch_kincaid']:.1f}")
            if scores["flesch_kincaid"] >= 70:
                st.success("Easy to read")
            elif scores["flesch_kincaid"] >= 50:
                st.warning("Moderate difficulty")
            else:
                st.error("Difficult to read")
        else:
            st.metric("Flesch-Kincaid", "N/A")
    
    with col2:
        if scores["gunning_fog"] is not None:
            st.metric("Gunning Fog", f"{scores['gunning_fog']:.1f}")
            if scores["gunning_fog"] <= 8:
                st.success("Simple text")
            elif scores["gunning_fog"] <= 12:
                st.warning("Moderate complexity")
            else:
                st.error("Complex text")
        else:
            st.metric("Gunning Fog", "N/A")
    
    with col3:
        if scores["smog_index"] is not None:
            st.metric("SMOG Index", f"{scores['smog_index']:.1f}")
            if scores["smog_index"] <= 8:
                st.success("Elementary level")
            elif scores["smog_index"] <= 12:
                st.warning("High school level")
            else:
                st.error("College level")
        else:
            st.metric("SMOG Index", "N/A")
    
    st.markdown("---")
    
    # Sentence distribution analysis
    st.subheader("📈 Sentence Distribution")
    results = classify_text(text)
    total = sum(results.values()) or 1

    st.write("**Sentence distribution**")
    st.write(f"- Beginner: {results['Beginner']}")
    st.write(f"- Intermediate: {results['Intermediate']}")
    st.write(f"- Advanced: {results['Advanced']}")

    df = pd.DataFrame({"count": [results["Beginner"], results["Intermediate"], results["Advanced"]]},
                      index=["Beginner", "Intermediate", "Advanced"])
    st.bar_chart(df)

    with st.expander("ℹ️ What do the levels mean?"):
        st.markdown("""
        **Readability Scores:**
        - **Flesch-Kincaid:** Higher score = easier text (70+ easy, 50-70 moderate, <50 difficult)
        - **Gunning Fog:** Higher score = more complex (≤8 simple, 8-12 moderate, ≥13 complex)
        - **SMOG Index:** Higher score = higher grade level needed (≤8 elementary, 9-12 high school, >12 college)
        
        **Sentence Distribution:**
        - **Beginner:** Favored by high FK (easier), low Gunning Fog, and low SMOG.  
        - **Intermediate:** Occurs at mid-range values for all three metrics.  
        - **Advanced:** Favored by low FK (harder), high Gunning Fog, and high SMOG.  
        
        The bars show the distribution of sentences:
        - Easier text → more sentences classified as **Beginner**.  
        - Moderate difficulty → more sentences classified as **Intermediate**.  
        - Higher difficulty → more sentences classified as **Advanced**.
        """)

def login_page():
    st.markdown("<div class='login-card'><div class='login-title'>🔐 Login</div>", unsafe_allow_html=True)
    with st.form("login_form", border=False):
        login = st.text_input("Email or Username", placeholder="user@example.com or username")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        remember = st.checkbox("Remember me")
        submit = st.form_submit_button("Sign In")
    if submit:
        db = SessionLocal()
        u = get_user_by_login(db, login.strip())
        if u and check_password(password, u.password_hash):
            token = create_jwt(u.username, hours=(24*7 if remember else 1))
            st.session_state["token"] = token
            st.session_state["username"] = u.username
            st.session_state["page"] = "main"
            st.rerun()
        else:
            st.error("Invalid credentials")
        db.close()
    if st.button("Create Account"):
        st.session_state["page"] = "signup"
    st.markdown("</div>", unsafe_allow_html=True)

def signup_page():
    st.markdown("<div class='login-card signup-card'><div class='login-title'>📝 Register</div>", unsafe_allow_html=True)
    with st.form("signup_form", border=False):
        email = st.text_input("Email")
        username = st.text_input("Username")
        name = st.text_input("Full Name")
        password = st.text_input("Password", type="password")
        language = st.radio("Preferred Language", ["English", "Hindi"], horizontal=True)
        age_group = st.selectbox("Age Group", ["<18", "18-25", "26-40", "40+"])
        submit = st.form_submit_button("Sign Up")
    if submit:
        if not email or not username or not password:
            st.error("Email, Username and Password required.")
        else:
            try:
                with SessionLocal() as db:
                    create_user(
                        db,
                        username=username,
                        email=email,
                        password=password,
                        name=name,
                        language=language,
                        age_group=age_group
                    )
                st.success("Account created! Please log in.")
                st.session_state["page"] = "login"
            except Exception:
                st.error("Username or email already exists.")

# ---------------- PROFILE / Dashboard ----------------
def preview_file(uploaded_file):
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()
    data = uploaded_file.read()
    bio = io.BytesIO(data)

    st.markdown("<div class='card'><div class='title'>👀 File Preview</div>", unsafe_allow_html=True)
    if ext == "txt":
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = data.decode("latin-1", errors="ignore")
        st.text_area("Text", text, height=240)
        return text, data
    elif ext == "csv":
        try:
            df = pd.read_csv(bio)
            st.dataframe(df)
            text = "\n".join(df.astype(str).fillna("").values.flatten())
            return text, data
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            return "", data
    elif ext in ("xlsx", "xls"):
        try:
            df = pd.read_excel(bio)
            st.dataframe(df)
            text = "\n".join(df.astype(str).fillna("").values.flatten())
            return text, data
        except Exception as e:
            st.error(f"Could not parse Excel: {e}")
            return "", data
    elif ext == "pdf":
        try:
            reader = PdfReader(bio)
            pages = min(5, len(reader.pages))
            text_pages = []
            for i in range(pages):
                t = reader.pages[i].extract_text() or ""
                text_pages.append(t)
            text = "\n".join(text_pages)
            st.text_area(f"PDF Text Preview (first {pages} pages)", text.strip(), height=260)
            return text, data
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            return "", data
    else:
        st.warning("Unsupported file type for preview")
        return "", data
    st.markdown("</div>", unsafe_allow_html=True)

def profile_page(username: str):
    ensure_uploads_table()
    with SessionLocal() as db:
        u = get_user_by_username(db, username)
        if not u:
            st.error("User not found.")
            return

    st.markdown(f"<div class='login-card'><div class='login-title'>👤 Welcome, {u.name or u.username}</div></div>", unsafe_allow_html=True)

    with st.expander("Update Profile"):
        with st.form("profile_form", border=False):
            name = st.text_input("Name", value=u.name or u.username)
            age_group = st.selectbox("Age Group", ["<18","18-25","26-40","40+"], index=["<18","18-25","26-40","40+"].index(u.age_group or "18-25"))
            language = st.radio("Language Preference", ["English", "Hindi"], index=0 if (u.language or "English")=="English" else 1, horizontal=True)
            save = st.form_submit_button("Save Profile")
        if save:
            with SessionLocal() as db2:
                update_user_profile(db2, username, name=name, language=language, age_group=age_group)
            st.success("Profile updated!")

    # Upload & analyze (accepts txt/csv/xlsx/pdf; analysis uses extracted text)
    uploaded_file = st.file_uploader("Drag and drop file here", type=["txt","csv","xlsx","pdf"])
    if uploaded_file:
        uploaded_file.seek(0)
        text, raw_bytes = preview_file(uploaded_file)
        
        # Create tabs for different analyses
        if text and text.strip():
            if SUMMARIZATION_AVAILABLE and PARAPHRASING_AVAILABLE:
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Readability Analysis", "📝 Text Summarization", "🔄 Text Paraphrasing", "📈 Combined Analysis"])
            elif SUMMARIZATION_AVAILABLE:
                tab1, tab2, tab3 = st.tabs(["📊 Readability Analysis", "📝 Text Summarization", "📈 Combined Analysis"])
            elif PARAPHRASING_AVAILABLE:
                tab1, tab2, tab3 = st.tabs(["📊 Readability Analysis", "🔄 Text Paraphrasing", "📈 Combined Analysis"])
            else:
                tab1, tab2 = st.tabs(["📊 Readability Analysis", "📈 Combined Analysis"])
            
            with tab1:
                # Run classification if we have text
                readability_level = None
                if text and text.strip():
                    # show analysis UI
                    readability_analysis_ui(text)
                    # derive dominant level (max count)
                    counts = classify_text(text)
                    if sum(counts.values()) > 0:
                        readability_level = max(counts, key=counts.get)
            
            if SUMMARIZATION_AVAILABLE:
                with tab2:
                    # Text summarization
                    summarize_text_ui(text)
                
                if PARAPHRASING_AVAILABLE:
                    with tab3:
                        # Text paraphrasing
                        paraphrase_text_ui(text)
                    
                    with tab4:
                        # Combined analysis view
                        st.subheader("📊 Complete Text Analysis")
                        
                        # Readability scores
                        scores = calculate_readability_scores(text)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if scores["flesch_kincaid"] is not None:
                                st.metric("Flesch-Kincaid", f"{scores['flesch_kincaid']:.1f}")
                        with col2:
                            if scores["gunning_fog"] is not None:
                                st.metric("Gunning Fog", f"{scores['gunning_fog']:.1f}")
                        with col3:
                            if scores["smog_index"] is not None:
                                st.metric("SMOG Index", f"{scores['smog_index']:.1f}")
                        
                        # Text statistics
                        st.subheader("📈 Text Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Words", len(text.split()))
                        with col2:
                            st.metric("Total Sentences", len(safe_sent_tokenize(text)))
                        with col3:
                            st.metric("Average Words per Sentence", round(len(text.split()) / len(safe_sent_tokenize(text)), 1))
                        with col4:
                            st.metric("Readability Level", readability_level or "N/A")
                else:
                    with tab3:
                        # Combined analysis view (without paraphrasing)
                        st.subheader("📊 Complete Text Analysis")
                        
                        # Readability scores
                        scores = calculate_readability_scores(text)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if scores["flesch_kincaid"] is not None:
                                st.metric("Flesch-Kincaid", f"{scores['flesch_kincaid']:.1f}")
                        with col2:
                            if scores["gunning_fog"] is not None:
                                st.metric("Gunning Fog", f"{scores['gunning_fog']:.1f}")
                        with col3:
                            if scores["smog_index"] is not None:
                                st.metric("SMOG Index", f"{scores['smog_index']:.1f}")
                        
                        # Text statistics
                        st.subheader("📈 Text Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Words", len(text.split()))
                        with col2:
                            st.metric("Total Sentences", len(safe_sent_tokenize(text)))
                        with col3:
                            st.metric("Average Words per Sentence", round(len(text.split()) / len(safe_sent_tokenize(text)), 1))
                        with col4:
                            st.metric("Readability Level", readability_level or "N/A")
            elif PARAPHRASING_AVAILABLE:
                with tab2:
                    # Text paraphrasing
                    paraphrase_text_ui(text)
                
                with tab3:
                    # Combined analysis view (without summarization)
                    st.subheader("📊 Complete Text Analysis")
                    
                    # Readability scores
                    scores = calculate_readability_scores(text)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if scores["flesch_kincaid"] is not None:
                            st.metric("Flesch-Kincaid", f"{scores['flesch_kincaid']:.1f}")
                    with col2:
                        if scores["gunning_fog"] is not None:
                            st.metric("Gunning Fog", f"{scores['gunning_fog']:.1f}")
                    with col3:
                        if scores["smog_index"] is not None:
                            st.metric("SMOG Index", f"{scores['smog_index']:.1f}")
                    
                    # Text statistics
                    st.subheader("📈 Text Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Words", len(text.split()))
                    with col2:
                        st.metric("Total Sentences", len(safe_sent_tokenize(text)))
                    with col3:
                        st.metric("Average Words per Sentence", round(len(text.split()) / len(safe_sent_tokenize(text)), 1))
                    with col4:
                        st.metric("Readability Level", readability_level or "N/A")
            else:
                with tab2:
                    # Combined analysis view (without summarization or paraphrasing)
                    st.subheader("📊 Complete Text Analysis")
                    
                    # Readability scores
                    scores = calculate_readability_scores(text)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if scores["flesch_kincaid"] is not None:
                            st.metric("Flesch-Kincaid", f"{scores['flesch_kincaid']:.1f}")
                    with col2:
                        if scores["gunning_fog"] is not None:
                            st.metric("Gunning Fog", f"{scores['gunning_fog']:.1f}")
                    with col3:
                        if scores["smog_index"] is not None:
                            st.metric("SMOG Index", f"{scores['smog_index']:.1f}")
                    
                    # Text statistics
                    st.subheader("📈 Text Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Words", len(text.split()))
                    with col2:
                        st.metric("Total Sentences", len(safe_sent_tokenize(text)))
                    with col3:
                        st.metric("Average Words per Sentence", round(len(text.split()) / len(safe_sent_tokenize(text)), 1))
                    with col4:
                        st.metric("Readability Level", readability_level or "N/A")
        
        # Save to DB with detected readability level (could be None)
        save_upload(username, uploaded_file.name, raw_bytes, readability_level)
        st.success(f"Saved {uploaded_file.name} (Readability: {readability_level or 'N/A'})")

    rows = list_uploads(username)
    st.markdown("<div class='login-card'><div class='login-title'>🗂 Your Uploads</div></div>", unsafe_allow_html=True)
    if rows:
        for fid, fname, level, created in rows:
            st.write(f"📄 {fname}  •  ID: {fid}  •  {created}  •  🔎 {level or 'N/A'}")
    else:
        st.info("No files uploaded yet.")

    if st.button("Logout"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.rerun()

# ---------------- AUTH helpers ----------------
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_jwt(username: str, hours=1):
    payload = {"username": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=hours)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded["username"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_user_by_login(db, login: str):
    stmt = select(User).where(User.email == login.lower()) if "@" in login else select(User).where(User.username == login.lower())
    return db.scalar(stmt)

def get_user_by_username(db, username: str):
    return db.scalar(select(User).where(User.username == username.lower()))

def create_user(db, username: str, email: str, password: str, name: str, language: str, age_group: str):
    u = User(
        username=username.lower(),
        email=email.lower(),
        name=name or username,
        password_hash=hash_password(password),
        language=language,
        age_group=age_group,
    )
    db.add(u)
    db.commit()
    return u

def update_user_profile(db, username: str, *, name: str, language: str, age_group: str):
    u = get_user_by_username(db, username)
    if not u:
        return None
    u.name = name
    u.language = language
    u.age_group = age_group
    db.commit()
    return u

# ---------------- MAIN APP ----------------
def main_app(username: str):
    profile_page(username)

def app():
    # ensure uploads table exists
    ensure_uploads_table()

    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    token = st.session_state.get("token")
    page = st.session_state["page"]

    if token:
        username = st.session_state.get("username")
        if not username:
            username = verify_jwt(token)
            if username:
                st.session_state["username"] = username
        if username:
            st.session_state["page"] = "main"
            main_app(username)
        else:
            st.session_state.clear()
            st.session_state["page"] = "login"
            login_page()
    else:
        if page == "login":
            login_page()
        elif page == "signup":
            signup_page()
        else:
            login_page()

if __name__ == "__main__":
    app()
