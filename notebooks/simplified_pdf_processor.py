"""
Simplified PDF Processor using pdfplumber (no Detectron2 required)
"""
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class PDFProcessor:
    """
    Handles PDF processing and region extraction using pdfplumber
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.plumber_doc = pdfplumber.open(pdf_path)

    def extract_page_image(self, page_num, dpi=300):
        """Extract page as high-resolution image"""
        page = self.doc[page_num]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img)

    def extract_regions(self, page_num):
        """Extract all regions from a page using pdfplumber"""
        image = self.extract_page_image(page_num)
        plumber_page = self.plumber_doc.pages[page_num]

        regions = []

        # Extract text-based regions using pdfplumber
        words = plumber_page.extract_words()

        if not words:
            # If no words found, create a single region for the whole page
            h, w = image.shape[:2]
            regions.append({
                'type': 'Text',
                'bbox': (0, 0, w, h),
                'image': image,
                'confidence': 0.5
            })
            return regions, image

        # Group words into text blocks based on vertical position
        # Sort by top position
        words = sorted(words, key=lambda x: (x['top'], x['x0']))

        # Simple clustering: group words that are vertically close
        current_block = []
        blocks = []
        y_threshold = 20  # pixels threshold for grouping

        for word in words:
            if not current_block:
                current_block.append(word)
            else:
                # Check if this word is close to the last word in current block
                last_word = current_block[-1]
                if abs(word['top'] - last_word['top']) < y_threshold:
                    current_block.append(word)
                else:
                    # Start new block
                    blocks.append(current_block)
                    current_block = [word]

        if current_block:
            blocks.append(current_block)

        # Convert blocks to regions
        scale_x = image.shape[1] / plumber_page.width
        scale_y = image.shape[0] / plumber_page.height

        for block in blocks:
            # Get bounding box of entire block
            x0 = min(w['x0'] for w in block)
            x1 = max(w['x1'] for w in block)
            y0 = min(w['top'] for w in block)
            y1 = max(w['bottom'] for w in block)

            # Scale to image coordinates
            x0_img = int(x0 * scale_x)
            x1_img = int(x1 * scale_x)
            y0_img = int(y0 * scale_y)
            y1_img = int(y1 * scale_y)

            # Ensure coordinates are within bounds
            x0_img = max(0, x0_img)
            y0_img = max(0, y0_img)
            x1_img = min(image.shape[1], x1_img)
            y1_img = min(image.shape[0], y1_img)

            # Extract region image
            if y1_img > y0_img and x1_img > x0_img:
                region_img = image[y0_img:y1_img, x0_img:x1_img]

                # Determine type based on size and position
                height = y1_img - y0_img
                width = x1_img - x0_img
                aspect_ratio = width / height if height > 0 else 0

                if height < 50:
                    region_type = 'Title'
                elif aspect_ratio > 3:
                    region_type = 'List'
                else:
                    region_type = 'Text'

                regions.append({
                    'type': region_type,
                    'bbox': (x0_img, y0_img, x1_img, y1_img),
                    'image': region_img,
                    'confidence': 0.8
                })

        # Also extract images/figures
        try:
            images_info = plumber_page.images
            for img_info in images_info:
                x0_img = int(img_info['x0'] * scale_x)
                x1_img = int(img_info['x1'] * scale_x)
                y0_img = int(img_info['top'] * scale_y)
                y1_img = int(img_info['bottom'] * scale_y)

                # Ensure coordinates are within bounds
                x0_img = max(0, x0_img)
                y0_img = max(0, y0_img)
                x1_img = min(image.shape[1], x1_img)
                y1_img = min(image.shape[0], y1_img)

                if y1_img > y0_img and x1_img > x0_img:
                    region_img = image[y0_img:y1_img, x0_img:x1_img]

                    regions.append({
                        'type': 'Figure',
                        'bbox': (x0_img, y0_img, x1_img, y1_img),
                        'image': region_img,
                        'confidence': 0.9
                    })
        except Exception as e:
            print(f"Warning: Could not extract images: {e}")

        return regions, image

    def visualize_layout(self, page_num, save_path=None):
        """Visualize detected layout"""
        regions, image = self.extract_regions(page_num)

        fig, ax = plt.subplots(figsize=(15, 20))
        ax.imshow(image)

        # Color map for different types
        colors = {
            'Text': 'red',
            'Title': 'blue',
            'List': 'green',
            'Table': 'purple',
            'Figure': 'orange'
        }

        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            color = colors.get(region['type'], 'red')

            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-10, f"{region['type']}: {region['confidence']:.2f}",
                   color=color, fontsize=8, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.axis('off')
        ax.set_title(f'Page {page_num + 1} - Layout Detection', fontsize=16, pad=20)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()

print("Simplified PDFProcessor loaded successfully!")
