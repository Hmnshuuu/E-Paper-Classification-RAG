"""
PDF Processor Module
Handles PDF processing, layout detection, and region extraction for e-papers
"""

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import layoutparser as lp
import matplotlib.pyplot as plt
import cv2


class PDFProcessor:
    """
    Processes PDF newspapers and extracts regions using layout detection.

    Addresses the challenge of identifying structural elements in digitized e-papers
    where traditional visual cues (borders, fonts, section tags) may be lost.
    """

    def __init__(self, pdf_path, detection_threshold=0.5):
        """
        Initialize PDF processor with layout detection model.

        Args:
            pdf_path: Path to the PDF file
            detection_threshold: Confidence threshold for layout detection (default: 0.5)
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

        # Initialize layout detection model (PubLayNet trained on scientific publications)
        # This model detects: Text, Title, List, Table, Figure
        self.layout_model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", detection_threshold],
            label_map={
                0: "Text",
                1: "Title",
                2: "List",
                3: "Table",
                4: "Figure"
            }
        )

    def extract_page_image(self, page_num, dpi=300):
        """
        Extract page as high-resolution image for better OCR accuracy.

        Args:
            page_num: Page number (0-indexed)
            dpi: Resolution for extraction (default: 300 for good OCR quality)

        Returns:
            numpy array of the page image
        """
        page = self.doc[page_num]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img)

    def detect_layout(self, image):
        """
        Detect layout elements in the page image.

        This addresses the challenge of losing structural cues in digitized e-papers
        by programmatically identifying different content regions.

        Args:
            image: Page image as numpy array

        Returns:
            Layout object containing detected regions
        """
        layout = self.layout_model.detect(image)
        return layout

    def extract_regions(self, page_num, min_region_size=100):
        """
        Extract all content regions from a page.

        Segments the page into individual regions that can be classified
        as news articles or advertisements.

        Args:
            page_num: Page number (0-indexed)
            min_region_size: Minimum region size in pixels to filter noise

        Returns:
            Tuple of (regions list, original page image)
        """
        image = self.extract_page_image(page_num)
        layout = self.detect_layout(image)

        regions = []
        for block in layout:
            x1, y1, x2, y2 = int(block.block.x_1), int(block.block.y_1), \
                             int(block.block.x_2), int(block.block.y_2)

            # Filter out tiny regions (likely noise)
            width = x2 - x1
            height = y2 - y1
            if width * height < min_region_size:
                continue

            region_img = image[y1:y2, x1:x2]

            regions.append({
                'type': block.type,
                'bbox': (x1, y1, x2, y2),
                'image': region_img,
                'confidence': block.score,
                'width': width,
                'height': height,
                'area': width * height
            })

        return regions, image

    def visualize_layout(self, page_num, save_path=None):
        """
        Visualize detected layout with bounding boxes.

        Useful for debugging and understanding how the system segments the page.

        Args:
            page_num: Page number (0-indexed)
            save_path: Optional path to save the visualization
        """
        image = self.extract_page_image(page_num)
        layout = self.detect_layout(image)

        fig, ax = plt.subplots(figsize=(15, 20))
        ax.imshow(image)

        for block in layout:
            x1, y1, x2, y2 = block.block.x_1, block.block.y_1, \
                             block.block.x_2, block.block.y_2

            rect = plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, edgecolor='red', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1-10,
                f"{block.type}: {block.score:.2f}",
                color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

        ax.axis('off')
        ax.set_title(f"Layout Detection - Page {page_num + 1}", fontsize=16, weight='bold')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()

    def get_page_count(self):
        """Get total number of pages in PDF"""
        return len(self.doc)

    def close(self):
        """Close the PDF document"""
        self.doc.close()
