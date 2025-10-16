# Lightweight configuration to prevent memory crashes

# Reduce image DPI (lower resolution = less memory)
LOW_MEMORY_DPI = 150  # Instead of 300

# Process one language at a time (reduce EasyOCR memory)
LANGUAGES_SINGLE = ['en']  # Start with English only

# Use smaller classification model
SMALL_CLASSIFIER_MODEL = 'typeform/distilbert-base-uncased-mnli'  # Instead of BART

# Batch size for processing
SMALL_BATCH_SIZE = 1  # Process regions one at a time

print("Lightweight configuration loaded")
print("This reduces memory usage by:")
print("  - Lower DPI (150 instead of 300)")
print("  - English-only OCR")
print("  - Smaller classification model (280MB vs 1.6GB)")
