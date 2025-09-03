@echo off
echo Installing required dependencies for Text Summarization AI...
echo.

echo Installing rouge-score...
pip install rouge-score

echo.
echo Installing accelerate...
pip install accelerate

echo.
echo Installing sentence-transformers...
pip install sentence-transformers

echo.
echo Installing transformers (if not already installed)...
pip install transformers

echo.
echo Downloading NLTK data...
python download_nltk_data.py

echo.
echo All dependencies installed successfully!
echo You can now run the application with: streamlit run app.py
pause
