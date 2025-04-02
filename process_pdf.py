import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(img):
    """Preprocess image for better OCR accuracy."""
    # Convert to grayscale
    img = img.convert("L")
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    # Reduce noise
    img = img.filter(ImageFilter.MedianFilter())
    return img

def extract_text_selectable(pdf_path):
    """Extract text from selectable PDFs."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error in selectable extraction: {e}")
    return text

def extract_text_ocr(pdf_path):
    """Extract text from non-selectable PDFs with improved OCR."""
    text = ""
    try:
        pdf = fitz.open(pdf_path)
        for page in pdf:
            pix = page.get_pixmap(dpi=300)  # Increase DPI for better resolution
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = preprocess_image(img)  # Preprocess image
            # Use English and Vietnamese languages
            text += pytesseract.image_to_string(img, lang="eng") + "\n"
        pdf.close()
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
    return text

def process_pdf(pdf_path):
    """Process a PDF, trying selectable text first, then OCR if needed."""
    text = extract_text_selectable(pdf_path)
    if not text.strip():  # If no text extracted, use OCR
        text = extract_text_ocr(pdf_path)
    return text