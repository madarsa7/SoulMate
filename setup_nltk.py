import nltk
import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_nltk_resources():
    """Download necessary NLTK resources and print status"""
    print("Downloading NLTK resources...")
    
    # Create a local directory for NLTK data
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add this directory to NLTK's search path
    nltk.data.path.insert(0, nltk_data_dir)
    
    resources = [
        'punkt',
        'stopwords'
    ]
    
    success = True
    
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
            print(f"✓ Successfully downloaded {resource}")
        except Exception as e:
            print(f"✗ Failed to download {resource}: {str(e)}")
            success = False
    
    print("\nVerifying downloads:")
    for resource in resources:
        try:
            resource_path = f"{'tokenizers/' if resource == 'punkt' else 'corpora/'}{resource}"
            nltk.data.find(resource_path)
            print(f"✓ {resource} is available at: {nltk.data.find(resource_path)}")
        except LookupError as e:
            print(f"✗ {resource} is NOT available: {str(e)}")
            success = False
    
    print("\nNLTK data directories:")
    for path in nltk.data.path:
        print(f"- {path}")
        if os.path.exists(path):
            print("  (directory exists)")
            # List contents if it's our custom directory
            if path == nltk_data_dir:
                print("  Contents:")
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        print(f"    - {os.path.join(root, d)}")
        else:
            print("  (directory does not exist)")
    
    return success

if __name__ == "__main__":
    success = download_nltk_resources()
    
    if success:
        print("\nNLTK resources were successfully downloaded and verified.")
        print("The application should now be able to run without NLTK-related errors.")
    else:
        print("\nWARNING: Some NLTK resources could not be downloaded or verified.")
        print("However, the application has been modified to handle missing NLTK resources gracefully.")
        print("You can still run the application, but some advanced text analysis features might use fallback methods.")