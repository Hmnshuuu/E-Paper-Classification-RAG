"""
Verification script to check if all dependencies are installed correctly
Run this after installation to verify everything works
"""

import sys

def check_import(package_name, import_name=None):
    """Check if a package can be imported"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} installed successfully")
        return True
    except ImportError as e:
        print(f"✗ {package_name} FAILED: {e}")
        return False

def main():
    print("="*60)
    print("E-Paper Classification System - Setup Verification")
    print("="*60)
    print()

    packages = [
        ("PyTorch", "torch"),
        ("torchvision", "torchvision"),
        ("PyMuPDF", "fitz"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("Transformers", "transformers"),
        ("EasyOCR", "easyocr"),
        ("Sentence-Transformers", "sentence_transformers"),
        ("LangChain", "langchain"),
        ("LangChain-Community", "langchain_community"),
        ("ChromaDB", "chromadb"),
        ("OpenCV", "cv2"),
        ("Pillow", "PIL"),
        ("Jupyter", "jupyter"),
    ]

    print("Checking core dependencies:\n")

    results = []
    for pkg_name, import_name in packages:
        results.append(check_import(pkg_name, import_name))

    print()
    print("="*60)
    success_count = sum(results)
    total_count = len(results)

    if success_count == total_count:
        print(f"✓ ALL {total_count} packages installed successfully!")
        print()
        print("You're ready to run the notebook!")
        print()
        print("Next steps:")
        print("  1. Run: jupyter notebook")
        print("  2. Open: notebooks/epaper_classification_rag.ipynb")
        print("  3. Execute all cells")
    else:
        print(f"⚠ {total_count - success_count} packages failed to install")
        print()
        print("Try reinstalling missing packages:")
        print("  pip install -r requirements.txt")

    print("="*60)

    # Additional info
    print()
    print("System Information:")
    print(f"  Python: {sys.version.split()[0]}")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
    except:
        pass

    print()

if __name__ == "__main__":
    main()
