"""
OCR Extractor Module
Handles multilingual text extraction from image regions
"""

import easyocr
import numpy as np
import torch


class OCRExtractor:
    """
    Multilingual OCR extractor using EasyOCR.

    Supports Indian newspapers published in various languages,
    addressing the multilingual requirement of the system.
    """

    def __init__(self, languages=['en', 'hi'], gpu=None):
        """
        Initialize EasyOCR reader with specified languages.

        Args:
            languages: List of language codes (e.g., ['en', 'hi', 'ta', 'te'])
                      Supported: en (English), hi (Hindi), ta (Tamil), te (Telugu),
                                 bn (Bengali), mr (Marathi), gu (Gujarati), etc.
            gpu: Whether to use GPU (None = auto-detect, True = force GPU, False = CPU only)
        """
        if gpu is None:
            gpu = torch.cuda.is_available()

        print(f"Initializing OCR with languages: {languages}")
        print(f"Using GPU: {gpu}")

        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.languages = languages

    def extract_text(self, image, detail=1):
        """
        Extract text from an image region.

        Args:
            image: Image region as numpy array
            detail: Level of detail (0 = text only, 1 = text + confidence + bbox)

        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        try:
            # Handle empty or invalid images
            if image is None or image.size == 0:
                return "", 0.0

            # Ensure image is in correct format
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)

            results = self.reader.readtext(image, detail=detail)

            if not results:
                return "", 0.0

            # Extract text and confidence
            if detail == 0:
                text = ' '.join(results)
                confidence = 1.0  # Not available in detail=0
            else:
                texts = []
                confidences = []

                for bbox, text, conf in results:
                    texts.append(text)
                    confidences.append(conf)

                text = ' '.join(texts)
                confidence = np.mean(confidences) if confidences else 0.0

            return text, confidence

        except Exception as e:
            print(f"OCR Error: {e}")
            return "", 0.0

    def extract_from_regions(self, regions, show_progress=True):
        """
        Extract text from multiple regions.

        Args:
            regions: List of region dictionaries containing 'image' key
            show_progress: Whether to print progress updates

        Returns:
            Updated regions list with 'text' and 'ocr_confidence' fields
        """
        total = len(regions)

        for i, region in enumerate(regions):
            if show_progress and (i + 1) % 5 == 0:
                print(f"  OCR progress: {i+1}/{total} regions")

            text, confidence = self.extract_text(region['image'])
            region['text'] = text
            region['ocr_confidence'] = confidence

            # Detect language (simple heuristic based on character sets)
            region['detected_language'] = self._detect_language(text)

        return regions

    def _detect_language(self, text):
        """
        Simple language detection based on character sets.

        Args:
            text: Input text string

        Returns:
            Detected language code ('en', 'hi', or 'mixed')
        """
        if not text:
            return 'unknown'

        # Check for Hindi (Devanagari script)
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')

        # Check for English
        english_count = sum(1 for char in text if 'a' <= char.lower() <= 'z')

        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return 'unknown'

        devanagari_ratio = devanagari_count / total_chars
        english_ratio = english_count / total_chars

        if devanagari_ratio > 0.3 and english_ratio > 0.3:
            return 'mixed'
        elif devanagari_ratio > english_ratio:
            return 'hi'
        elif english_ratio > devanagari_ratio:
            return 'en'
        else:
            return 'unknown'

    def add_language_support(self, new_languages):
        """
        Add support for additional languages (requires re-initialization).

        Args:
            new_languages: List of new language codes to add
        """
        updated_languages = list(set(self.languages + new_languages))
        self.__init__(languages=updated_languages)
