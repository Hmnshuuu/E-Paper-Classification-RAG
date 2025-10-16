# Quick Start Guide

## Installation

### ✅ Recommended: Using Virtual Environment (venv)

**Step 1: Create virtual environment**
```bash
python -m venv venv
```

**Step 2: Activate the environment**
```bash
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

### Alternative: Development Installation

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install package in development mode
pip install -e .
```

## Files Explained

- **`requirements.txt`** - All required Python packages
- **`setup.py`** - Python package setup for development mode
- **`venv/`** - Virtual environment directory (auto-generated)

## Running the Project

1. **Activate the environment:**
   ```bash
   # Windows:
   venv\Scripts\activate

   # Linux/Mac:
   source venv/bin/activate
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the main notebook:**
   - Navigate to `notebooks/epaper_classification_rag.ipynb`
   - Run cells sequentially

4. **Process your PDF:**
   - Your PDF is already placed: `data/raw/TOI_Delhi_14-10-2025.pdf`
   - The notebook is configured to process pages 1, 2, 9, 10
   - Just run the cells!

## Expected Output

After running the notebook, you'll get:

- **Layout visualizations** in `outputs/` directory
- **Classification results** as CSV file
- **Analysis charts** showing distribution of news vs ads
- **RAG system** ready for querying historical articles

## Troubleshooting

**Problem: CUDA/GPU errors**
- Solution: The system works on CPU, just slower
- Models will auto-detect and use CPU

**Problem: LayoutParser model download fails**
- Solution: Ensure internet connection, model auto-downloads on first run (~100MB)

**Problem: EasyOCR language models missing**
- Solution: Will auto-download on first use (English ~40MB, Hindi ~20MB)

**Problem: Memory errors**
- Solution: Reduce DPI in configuration or process one page at a time

## Next Steps

After setup:
1. ✅ Run the notebook cells sequentially
2. ✅ Check `outputs/` for visualizations
3. ✅ Test RAG queries with different topics
4. ✅ Export results to CSV for analysis
5. ✅ Customize for your specific use case

## Support

Check the main README.md for detailed documentation.
