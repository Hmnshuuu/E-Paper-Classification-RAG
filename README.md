# E-Paper Content Classification & RAG System

A complete machine learning system for processing newspaper PDFs, classifying content (News vs Advertisements), and implementing a RAG system for information retrieval.

## Project Overview

This project implements:
- PDF processing and layout detection
- Multilingual OCR (English + Hindi)
- Content classification using transformers
- RAG system for historical article retrieval
- Automated summarization and metadata generation

## Project Structure

```
assignment/
├── data/
│   ├── raw/              # Place your PDF files here
│   ├── processed/        # Processed data outputs
│   └── chroma_db/        # Vector database storage
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks
│   └── epaper_classification_rag.ipynb
├── outputs/              # Visualizations and results
├── src/
│   ├── classification/   # Classification modules
│   ├── rag/             # RAG implementation
│   └── preprocessing/   # PDF and OCR processing
├── environment.yml       # Conda environment config
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8+ installed
- ~2GB disk space for dependencies
- Internet connection for downloading models

### Installation Steps

1. **Navigate to the project directory**
```bash
cd assignment
```

2. **Create Virtual Environment**
```bash
python -m venv venv
```

3. **Activate the environment**
```bash
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

Or simply double-click **`activate.bat`** on Windows!

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Usage

### 1. Prepare Your Data
Place your newspaper PDF files in the `data/raw/` directory.

### 2. Launch Jupyter Notebook
```bash
jupyter notebook
```

Navigate to `notebooks/epaper_classification_rag.ipynb`

### 3. Run the Notebook
Execute cells sequentially:
1. **Setup and Imports**: Load all required libraries
2. **Configuration**: Set paths and parameters
3. **PDF Processing**: Extract pages and detect layout
4. **OCR**: Extract multilingual text
5. **Classification**: Classify content as News/Ads
6. **RAG System**: Build vector index for retrieval
7. **Visualization**: Analyze results
8. **Query**: Test the RAG system with queries

### Example Code

```python
# Initialize pipeline
pdf_path = '../data/raw/your_newspaper.pdf'
pipeline = EPaperPipeline(pdf_path)

# Process all pages
pipeline.process_all_pages(visualize_first=True)

# Build RAG index
pipeline.build_rag_index()

# Query the system
results, summary = pipeline.query_rag("What are the latest political news?", k=5)
print(summary)
```

## Key Components

### 1. PDF Processor
- Extracts high-resolution page images
- Detects layout elements using LayoutParser
- Segments regions for classification

### 2. OCR Extractor
- Multilingual text extraction (English + Hindi)
- Uses EasyOCR for accuracy
- Provides confidence scores

### 3. Content Classifier
- Zero-shot classification using BART
- Distinguishes news articles from advertisements
- Feature-based classification fallback

### 4. RAG System
- Vector-based document retrieval using ChromaDB
- Semantic search with sentence transformers
- Extractive summarization
- Metadata filtering

## Technical Stack

- **PDF Processing**: PyMuPDF, pdfplumber
- **Layout Detection**: LayoutParser with Detectron2
- **OCR**: EasyOCR (supports 80+ languages)
- **ML/NLP**:
  - Transformers (HuggingFace)
  - Sentence-Transformers
  - PyTorch
- **RAG**:
  - LangChain
  - ChromaDB
  - HuggingFace Embeddings
- **Visualization**: Matplotlib, Seaborn

## Features

✅ Multilingual support (English, Hindi)
✅ High-accuracy OCR with confidence scores
✅ Automated content classification
✅ Semantic search and retrieval
✅ Visual layout analysis
✅ Extractive summarization
✅ Export results to CSV
✅ Comprehensive visualizations

## Performance Notes

- **GPU Acceleration**: Automatically uses CUDA if available
- **Processing Time**: ~30-60 seconds per page (depends on complexity)
- **Memory Requirements**: 4GB+ RAM recommended
- **PDF Quality**: Higher DPI = better OCR accuracy

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   - The system works on CPU, but slower
   - Check CUDA compatibility: `torch.cuda.is_available()`

2. **LayoutParser Model Download**
   - First run downloads ~100MB model
   - Requires internet connection

3. **OCR Language Support**
   - EasyOCR downloads language models on first use
   - Add more languages: `OCRExtractor(languages=['en', 'hi', 'ta'])`

4. **Memory Errors**
   - Process pages individually instead of batch
   - Reduce image DPI in configuration

### Getting Help

If you encounter issues:
1. Check the Jupyter notebook output for error messages
2. Verify all dependencies are installed: `pip list`
3. Ensure PDF files are not corrupted
4. Check GPU memory: `nvidia-smi` (if using CUDA)

## Output Files

After processing, check the `outputs/` directory for:
- `page_0_layout.png`: Visual layout detection
- `analysis_results.png`: Statistical visualizations
- `classification_results.csv`: Exported results

## Future Enhancements

- [ ] Fine-tune classification model on newspaper data
- [ ] Implement abstractive summarization
- [ ] Add named entity recognition
- [ ] Support for more Indian languages (Tamil, Telugu, etc.)
- [ ] Web interface for easy access
- [ ] Batch processing optimization
- [ ] Real-time processing pipeline

## Timeline

**Deadline**: Thursday 5 PM

**Milestones**:
- ✅ Project setup and environment
- ✅ Notebook template created
- [ ] Test with sample PDFs
- [ ] Optimize classification accuracy
- [ ] Complete RAG implementation
- [ ] Documentation and visualization
- [ ] Final testing and submission

## License

This project is created for an ML assignment/interview.

## Contact

For questions or issues, refer to the notebook documentation or check the inline comments in the code.

---

**Good luck with your interview! 🚀**
