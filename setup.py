"""
Setup script for E-Paper Classification & RAG System
Run this to install the package and all dependencies
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="epaper-classification",
    version="1.0.0",
    author="Your Name",
    description="E-Paper Content Classification and RAG System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "PyMuPDF>=1.23.8",
        "layoutparser[paddleocr]>=0.3.4",
        "easyocr>=1.7.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "sentence-transformers>=2.2.2",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "chromadb>=0.4.22",
        "tiktoken>=0.5.2",
        "Pillow>=10.1.0",
        "opencv-python>=4.9.0",
        "pdfplumber>=0.10.3",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "scikit-learn>=1.3.0",
        "jupyter>=1.0.0",
        "notebook>=7.0.0",
        "ipykernel>=6.27.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
