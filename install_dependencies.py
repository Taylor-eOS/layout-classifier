import os
import subprocess
import sys

packages = {
    "fitz": "PyMuPDF",
    "wordfreq": "wordfreq",
    "pygame": "pygame",
    "numpy": "numpy",
    "tensorflow": "tensorflow",
    "sklearn": "scikit-learn",
    "spacy": "spacy",
    "joblib": "joblib",
}

for module_name, package_name in packages.items():
    try:
        __import__(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
