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
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_ocr(pdf_path):
    text = ""
    pdf = fitz.open(pdf_path)
    for page in pdf:
        pix = page.get_pixmap(dpi=150)  # Lower DPI for speed (adjust as needed)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img) + "\n"
    pdf.close()
    return text

def process_pdf(pdf_path):
    text = extract_text_selectable(pdf_path)
    # Only use OCR if selectable text is empty or too short
    if not text.strip() or len(text) < 50:
        text = extract_text_ocr(pdf_path)
    return text