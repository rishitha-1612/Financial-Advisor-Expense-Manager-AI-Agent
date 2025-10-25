import cv2
import pytesseract
import re
import os
import pandas as pd
from datetime import datetime
import sys
import logging

# ---------- CONFIG ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_FOLDER, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_FOLDER, "expenses.csv")

# ---------- CATEGORY MAPPING ----------
CATEGORY_KEYWORDS = {
    "Food": ["restaurant", "cafe", "coffee", "pizza", "burger", "meal", "dining"],
    "Transport": ["uber", "ola", "taxi", "bus", "fuel", "petrol", "metro"],
    "Groceries": ["supermarket", "grocery", "walmart", "costco", "store"],
    "Entertainment": ["movie", "netflix", "spotify", "theatre", "concert"],
    "Shopping": ["amazon", "flipkart", "mall", "shop", "clothes", "store"],
    "Health": ["pharmacy", "hospital", "clinic", "doctor"],
    "Utilities": ["electricity", "water", "internet", "bill"],
}

def categorize_expense(vendor, description=""):
    """Categorize expense based on vendor name or description."""
    text = (vendor + " " + description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    return "Other"

def financial_advice(amount, category):
    """Generate a simple advice based on amount and category."""
    try:
        amt = float(amount)
    except ValueError:
        return "No advice available"

    if category == "Food" and amt > 1000:
        return "Consider reducing dining expenses."
    elif category == "Transport" and amt > 500:
        return "Consider using public transport to save money."
    elif category == "Entertainment" and amt > 1000:
        return "Track entertainment expenses to avoid overspending."
    elif category == "Other" and amt > 2000:
        return "High miscellaneous expense. Review this purchase."
    else:
        return "Expense looks normal."

# ---------- VALIDATION ----------
def validate_image_path(image_path):
    """Validate that the image path exists and is readable."""
    if not image_path:
        raise ValueError("Image path is empty")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    if os.path.splitext(image_path)[1].lower() not in valid_extensions:
        raise ValueError(f"Unsupported file format. Supported: {valid_extensions}")

# ---------- IMAGE PREPROCESS ----------
def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        scale_percent = 150
        width = int(thresh.shape[1] * scale_percent / 100)
        height = int(thresh.shape[0] * scale_percent / 100)
        resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_CUBIC)
        return resized
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

# ---------- CLEAN TEXT ----------
def clean_text(text):
    """Clean OCR text by removing unwanted characters."""
    text = re.sub(r"[^\w₹$€£.,:/\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- EXTRACT AMOUNT ----------
def extract_amount(text):
    """Extract transaction amount with multi-currency support."""
    amount = "Not found"
    text_lower = text.lower()
    
    currency_pattern = r"(?:₹|\$|€|£)\s*([\d,]+(?:\.\d{1,2})?)"
    for match in re.finditer(currency_pattern, text):
        amt_str = match.group(1).replace(",", "").strip()
        if validate_amount(amt_str):
            return amt_str
    
    paid_pattern = r"(?:paid|amount|total)[\s:]*(\d[\d,]*(?:\.\d{1,2})?)"
    paid_match = re.search(paid_pattern, text_lower)
    if paid_match:
        amt_str = paid_match.group(1).replace(",", "").strip()
        if validate_amount(amt_str):
            return amt_str
    
    all_numbers = re.findall(r"\d[\d,]*(?:\.\d{1,2})?", text)
    candidates = [float(num.replace(",", "")) for num in all_numbers if validate_amount(num.replace(",", ""))]
    if candidates:
        candidates.sort()
        return str(candidates[0])
    
    return amount

def validate_amount(amt_str):
    try:
        val = float(amt_str)
        return 5 <= val <= 1000000
    except ValueError:
        return False

# ---------- EXTRACT VENDOR ----------
def extract_vendor(text):
    vendor = "Not found"
    lines = text.split('\n')
    keyword_patterns = [
        r"(?:to|pay)\s+([a-zA-Z\s&'-]+?)(?:\n|amount|paid|edit|share|$)",
        r"(?:merchant|store|shop)[\s:]*([a-zA-Z\s&'-]+?)(?:\n|amount|$)",
        r"([A-Z][a-zA-Z\s&'-]{3,})(?:\s+(?:store|café|mart|shop|ltd|pvt|inc))"
    ]
    for pattern in keyword_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            vendor_name = match.group(1).strip().title()
            vendor_name = re.sub(r'\b(pay|again|edit|share|amount|paid|to)\b', '', vendor_name, flags=re.IGNORECASE).strip()
            if 2 < len(vendor_name) < 100:
                return vendor_name
    
    for line in lines[:5]:
        line_clean = line.strip()
        if 3 < len(line_clean) < 100:
            uppercase_count = sum(1 for c in line_clean if c.isupper())
            if uppercase_count >= len(line_clean) * 0.4 and not any(x in line_clean.lower() for x in ['ave','st','tx','ca','rd','blvd','open','close','am','pm','phone']):
                vendor_name = line_clean.replace('~','').replace('_','').replace('|','').strip()
                if len(vendor_name) > 3:
                    return vendor_name
    
    store_patterns = [r"(trader\s+joes?|costco|walmart|target|whole\s+foods?|kroger|safeway|albertsons|publix)"]
    for pattern in store_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).title()
    
    return vendor

# ---------- EXTRACT DATE ----------
def extract_date(text):
    date = "Not found"
    patterns = [
        r"(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})",
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"(?:date|on|at)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return date

# ---------- EXTRACT DETAILS ----------
def extract_details(text):
    text = clean_text(text)
    amount = extract_amount(text)
    vendor = extract_vendor(text)
    date = extract_date(text)
    category = categorize_expense(vendor, text)
    advice = financial_advice(amount, category)
    return amount, vendor, date, category, advice

# ---------- PROCESS IMAGE ----------
def extract_expenses(image_path):
    try:
        validate_image_path(image_path)
        logger.info(f"Processing image: {image_path}")
        
        processed_img = preprocess_image(image_path)
        custom_config = r'--psm 6 --oem 3 -l eng'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        if not text.strip():
            raise ValueError("No text detected in image")
        
        amount, vendor, date, category, advice = extract_details(text)
        
        current_entry = pd.DataFrame([{
            "File": os.path.basename(image_path),
            "Amount": amount,
            "Vendor": vendor,
            "Category": category,
            "Advice": advice,
            "Date": date,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        current_entry.to_csv(OUTPUT_CSV, index=False, quotechar='"', quoting=1)
        logger.info(f"CSV saved: {OUTPUT_CSV}")
        
        print("CSV generated\n")
        print("RAW_OUTPUT_START")
        print(text)
        print("RAW_OUTPUT_END")
        
        print("\nExtracted Fields:")
        print(f"Amount   : {amount}")
        print(f"Vendor   : {vendor}")
        print(f"Category : {category}")
        print(f"Advice   : {advice}")
        print(f"Date     : {date}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: No image path provided.", file=sys.stderr)
        sys.exit(1)
    image_path = sys.argv[1]
    extract_expenses(image_path)
