"""
Content Classifier Module
Classifies newspaper regions as News Articles or Advertisements
"""

import torch
from transformers import AutoTokenizer, pipeline
import numpy as np


class ContentClassifier:
    """
    Classifies content regions as News Articles (📰) or Advertisements (📢).

    Uses a combination of:
    1. Text-based classification using zero-shot learning
    2. Visual/structural feature-based heuristics
    3. Position and layout analysis

    This addresses the challenge of distinguishing news from ads when
    traditional visual cues are lost in digitized e-papers.
    """

    def __init__(self, model_name='facebook/bart-large-mnli', use_gpu=None):
        """
        Initialize the classifier.

        Args:
            model_name: HuggingFace model for zero-shot classification
            use_gpu: Whether to use GPU (None = auto-detect)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        print(f"Initializing classifier on device: {self.device}")

        # Zero-shot classification pipeline
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if use_gpu else -1
            )
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load classification model: {e}")
            print("Falling back to heuristic-based classification only")
            self.model_loaded = False

        # Classification labels
        self.labels = ["news article", "advertisement"]

        # Advertisement keywords (multilingual)
        self.ad_keywords = [
            'sale', 'offer', 'discount', 'buy', 'deal', 'price', 'shop',
            'call', 'contact', 'visit', 'limited', 'offer', 'free',
            'बिक्री', 'छूट', 'खरीदें', 'मुफ्त',  # Hindi
            'promo', 'exclusive', 'now', 'today', 'hurry'
        ]

        # News keywords
        self.news_keywords = [
            'said', 'according', 'report', 'government', 'minister', 'president',
            'court', 'police', 'official', 'announced', 'statement',
            'सरकार', 'मंत्री', 'पुलिस',  # Hindi
            'election', 'meeting', 'conference'
        ]

    def extract_features(self, region):
        """
        Extract visual and structural features from a region.

        These features help identify advertisements which often have:
        - Different aspect ratios (square, wide banners)
        - Specific positions (top/bottom of page, sidebars)
        - Images with minimal text
        - Smaller or larger than typical article blocks

        Args:
            region: Region dictionary with bbox, type, text, etc.

        Returns:
            Dictionary of extracted features
        """
        text = region.get('text', '')
        bbox = region.get('bbox', (0, 0, 100, 100))
        width = region.get('width', 100)
        height = region.get('height', 100)

        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_image': region['type'] == 'Figure',
            'is_title': region['type'] == 'Title',
            'aspect_ratio': self._calculate_aspect_ratio(bbox),
            'position': self._get_position_category(bbox),
            'area': width * height,
            'is_large': (width * height) > 500000,  # Large regions
            'is_small': (width * height) < 10000,   # Small regions
        }

        # Text-based features
        if text:
            text_lower = text.lower()
            features['ad_keyword_count'] = sum(
                1 for kw in self.ad_keywords if kw in text_lower
            )
            features['news_keyword_count'] = sum(
                1 for kw in self.news_keywords if kw in text_lower
            )
            features['has_price'] = any(c in text for c in ['₹', '$', 'Rs', 'INR'])
            features['has_contact'] = any(w in text_lower for w in ['call', 'contact', 'email', 'phone'])
        else:
            features['ad_keyword_count'] = 0
            features['news_keyword_count'] = 0
            features['has_price'] = False
            features['has_contact'] = False

        return features

    def _calculate_aspect_ratio(self, bbox):
        """Calculate width/height ratio"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width / height if height > 0 else 1.0

    def _get_position_category(self, bbox):
        """Categorize position on page (top/middle/bottom)"""
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2

        if center_y < 1000:
            return 'top'
        elif center_y < 2000:
            return 'middle'
        else:
            return 'bottom'

    def classify_by_heuristics(self, region, features):
        """
        Classify using heuristic rules based on visual and text features.

        Args:
            region: Region dictionary
            features: Extracted features

        Returns:
            Tuple of (label, confidence_score)
        """
        score = 0.5  # Neutral starting point

        # Strong advertisement indicators
        if features['has_price'] and features['has_contact']:
            score += 0.3
        if features['ad_keyword_count'] > 2:
            score += 0.2
        if features['is_small'] and features['has_image']:
            score += 0.15

        # Strong news indicators
        if features['news_keyword_count'] > 2:
            score -= 0.25
        if features['is_title'] and features['word_count'] > 5:
            score -= 0.2
        if features['text_length'] > 200 and not features['has_price']:
            score -= 0.15

        # Position-based (ads often at top/bottom)
        if features['position'] in ['top', 'bottom'] and features['is_small']:
            score += 0.1

        # Clamp score between 0 and 1
        score = max(0.0, min(1.0, score))

        # Determine label
        if score > 0.5:
            return 'advertisement', score
        else:
            return 'news article', 1.0 - score

    def classify_by_text(self, text):
        """
        Classify using NLP model (zero-shot classification).

        Args:
            text: Text content to classify

        Returns:
            Tuple of (label, confidence_score)
        """
        if not self.model_loaded:
            return None, 0.0

        try:
            # Truncate long text to fit model's max length
            text_truncated = text[:512]

            result = self.classifier(text_truncated, self.labels)
            label = result['labels'][0]
            score = result['scores'][0]

            return label, score

        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0.0

    def classify_region(self, region):
        """
        Classify a single region as news or advertisement.

        Uses ensemble approach:
        1. If sufficient text: Use NLP model
        2. If low text or model unavailable: Use heuristics
        3. Combine both if available

        Args:
            region: Region dictionary

        Returns:
            Tuple of (label, confidence_score)
        """
        text = region.get('text', '')
        features = self.extract_features(region)

        # Strategy 1: Text-based classification (if enough text)
        if len(text) > 20 and self.model_loaded:
            label_text, score_text = self.classify_by_text(text)

            if label_text and score_text > 0.7:
                # High confidence from text model
                return label_text, score_text
            elif label_text:
                # Moderate confidence - ensemble with heuristics
                label_heur, score_heur = self.classify_by_heuristics(region, features)

                # Weighted average
                combined_score = 0.7 * score_text + 0.3 * score_heur
                final_label = label_text if score_text > score_heur else label_heur

                return final_label, combined_score

        # Strategy 2: Heuristic-based (fallback or low text)
        return self.classify_by_heuristics(region, features)

    def classify_regions(self, regions, show_progress=True):
        """
        Classify multiple regions.

        Args:
            regions: List of region dictionaries
            show_progress: Whether to print progress

        Returns:
            Updated regions with 'classification' and 'classification_confidence'
        """
        total = len(regions)

        for i, region in enumerate(regions):
            if show_progress and (i + 1) % 5 == 0:
                print(f"  Classification progress: {i+1}/{total} regions")

            label, confidence = self.classify_region(region)
            region['classification'] = label
            region['classification_confidence'] = confidence

        return regions

    def get_classification_summary(self, regions):
        """
        Get summary statistics of classifications.

        Args:
            regions: List of classified regions

        Returns:
            Dictionary with classification statistics
        """
        news_count = sum(1 for r in regions if r.get('classification') == 'news article')
        ad_count = sum(1 for r in regions if r.get('classification') == 'advertisement')

        confidences = [r.get('classification_confidence', 0) for r in regions]

        return {
            'total_regions': len(regions),
            'news_articles': news_count,
            'advertisements': ad_count,
            'news_percentage': (news_count / len(regions) * 100) if regions else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
        }
