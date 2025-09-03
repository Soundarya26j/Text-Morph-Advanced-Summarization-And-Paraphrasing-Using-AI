#!/usr/bin/env python3
"""
Script to download NLTK data required for the text analysis application.
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    
    try:
        # Download punkt tokenizer
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=False)
        
        # Download punkt_tab (if available)
        try:
            print("Downloading punkt_tab...")
            nltk.download('punkt_tab', quiet=False)
        except Exception as e:
            print(f"Warning: punkt_tab not available: {e}")
        
        # Download tokenizers/punkt
        try:
            print("Downloading tokenizers/punkt...")
            nltk.download('tokenizers/punkt', quiet=False)
        except Exception as e:
            print(f"Warning: tokenizers/punkt not available: {e}")
        
        print("NLTK data download completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)
