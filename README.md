# Text Summarization AI with Readability Analysis

A comprehensive text analysis application that combines readability assessment with AI-powered summarization and paraphrasing using state-of-the-art models.

## Features

### 📊 Readability Analysis
- **Flesch-Kincaid Score**: Measures text difficulty (higher = easier)
- **Gunning Fog Index**: Measures complexity (lower = simpler)
- **SMOG Index**: Estimates grade level needed
- **Sentence Distribution**: Classifies sentences as Beginner/Intermediate/Advanced

### 📝 AI Text Summarization
- **Multiple Models**: PEGASUS, FLAN-T5, and BART
- **Length Options**: Short (~50 words), Medium (~150 words), Long (~300 words)
- **ROUGE Evaluation**: Automatic quality assessment with precision, recall, and F1 scores

### 🔄 AI Text Paraphrasing
- **Multiple Models**: FLAN-T5 and BART for paraphrasing
- **Complexity Levels**: Simplified, Standard, Enhanced, Academic
- **Paraphrasing Levels**: Sentence, Paragraph, and Full Text
- **Semantic Similarity**: Automatic quality assessment using sentence transformers
- **Side-by-Side Comparison**: Original vs paraphrased text display

### 📁 File Support
- **Text Files**: TXT, CSV, Excel (XLSX/XLS), PDF
- **User Management**: Registration, login, profile management
- **File Storage**: Secure database storage with metadata

## Installation

### Option 1: Automatic Installation (Windows)
1. Double-click `install_dependencies.bat`
2. Wait for installation to complete (includes NLTK data download)
3. Run the application: `streamlit run app.py`

### Option 2: Manual Installation
```bash
# Install all required packages
pip install -r requirements.txt

# Download NLTK data
python download_nltk_data.py

# Or install individually
pip install rouge-score accelerate sentence-transformers transformers
```

### Option 3: Using the Application Without AI Features
If you encounter issues with AI dependencies, the application will still work with:
- Readability analysis
- Text statistics
- File upload and storage

## Usage

1. **Start the application**: `streamlit run app.py`
2. **Register/Login**: Create an account or sign in
3. **Upload a document**: Drag and drop TXT, CSV, Excel, or PDF files
4. **Analyze text**: Use the tabs to explore different analyses
5. **Generate summaries**: Choose model and length for AI summarization
6. **Paraphrase text**: Select model, complexity level, and paraphrasing level

## Models Used

### Summarization Models
- **PEGASUS**: Excellent for news articles and factual content
- **FLAN-T5**: Versatile and accurate for various text types  
- **BART**: Great for general text summarization

### Paraphrasing Models
- **FLAN-T5**: Good for general paraphrasing with prompt engineering
- **BART**: Good for creative rewording and style changes

### Readability Metrics
- **Flesch-Kincaid**: 70+ (easy), 50-70 (moderate), <50 (difficult)
- **Gunning Fog**: ≤8 (simple), 8-12 (moderate), ≥13 (complex)
- **SMOG Index**: ≤8 (elementary), 9-12 (high school), >12 (college)

### Complexity Levels for Paraphrasing
- **Simplified**: Uses simpler vocabulary and shorter sentences
- **Standard**: Maintains original complexity level
- **Enhanced**: Uses more sophisticated vocabulary and complex structures
- **Academic**: Formal academic style with advanced terminology

## Troubleshooting

### Missing Dependencies
If you see "AI features are not available":
1. Run `pip install rouge-score accelerate sentence-transformers`
2. Restart the application

### NLTK Data Issues
If you see "Resource punkt_tab not found":
1. Run `python download_nltk_data.py`
2. Or manually run: `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"`
3. Restart the application

### Model Loading Issues
- Models are downloaded automatically on first use
- Ensure stable internet connection for initial download
- GPU acceleration is used if available, falls back to CPU

## Technical Details

- **Framework**: Streamlit
- **Database**: SQLite
- **AI Models**: Hugging Face Transformers
- **Evaluation**: ROUGE Score, Sentence Transformers
- **Authentication**: JWT tokens with bcrypt hashing

## License

This project is for educational and research purposes.
