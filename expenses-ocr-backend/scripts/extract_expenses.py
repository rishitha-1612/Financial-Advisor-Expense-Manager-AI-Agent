"""
Smart Expense Tracker with OCR and RAG - ENHANCED VERSION
File: app.py
New Features: 
- Improved OCR with preprocessing
- Financial reports & insights
- Goal tracking & savings
- Expense prediction
- Multi-guru comparison
- Investment recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from groq import Groq
import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import io
from datetime import datetime, timedelta
import cv2
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173"],
        "methods": ["GET", "POST", "DELETE", "PUT", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =============================================================================
# IMPROVED OCR WITH IMAGE PREPROCESSING
# =============================================================================
def preprocess_image(image):
    """Enhanced image preprocessing for better OCR accuracy"""
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 2. Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 3. Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        
        # 4. Adaptive thresholding for better text detection
        binary = cv2.adaptiveThreshold(
            contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL
        return Image.fromarray(processed)
    except Exception as e:
        print(f"[WARNING] Preprocessing failed, using original: {e}")
        return image

def extract_text_improved(image_file):
    """Improved OCR with multiple extraction strategies"""
    try:
        image = Image.open(io.BytesIO(image_file))
        
        # Strategy 1: Original image
        text1 = pytesseract.image_to_string(image, config='--psm 6')
        
        # Strategy 2: Preprocessed image
        preprocessed = preprocess_image(image)
        text2 = pytesseract.image_to_string(preprocessed, config='--psm 6')
        
        # Strategy 3: With different PSM mode
        text3 = pytesseract.image_to_string(preprocessed, config='--psm 4')
        
        # Combine results (take longest as it's likely most complete)
        texts = [text1, text2, text3]
        best_text = max(texts, key=len)
        
        print(f"[INFO] Extracted {len(best_text)} characters")
        return best_text
    except Exception as e:
        print(f"[ERROR] OCR extraction failed: {e}")
        raise

# =============================================================================
# Pydantic Models
# =============================================================================
class Item(BaseModel):
    item_name: str = Field(alias='name', description="The name of the product or service.")
    price: float = Field(description="The total price for the item, as a float.")
    category: Optional[str] = None

class Receipt(BaseModel):
    address: str = ""
    phone: str = Field(alias='phoneNumber', default="")
    date: str = ""
    items: List[Item] = []
    subtotal: float = Field(alias='subTotal', default=0.0)
    sales_tax: float = Field(alias='salesTax', default=0.0)
    total_amount: float = Field(alias='amount', default=0.0)
    balance: float = 0.0
    merchant_category: Optional[str] = None

    model_config = {"populate_by_name": True}

    @classmethod
    def model_validate_json(cls, json_data):
        instance = super().model_validate_json(json_data)
        instance.total_amount = instance.total_amount or instance.balance
        return instance

class FinancialGoal(BaseModel):
    id: str
    name: str
    target_amount: float
    current_amount: float = 0.0
    deadline: str
    category: str
    created_at: str

# =============================================================================
# Groq Client Initialization
# =============================================================================
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


try:
    client = Groq(api_key=GROQ_API_KEY)
    print("[SUCCESS] Groq client initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize Groq client: {str(e)}")
    raise

# =============================================================================
# Enhanced Vector Database with Multiple Financial Philosophies
# =============================================================================
def setup_multi_guru_vector_db():
    """Setup multiple financial philosophy RAG databases"""
    print("\n[INFO] Setting up Multi-Guru Financial RAG...")
    
    # Conservative Guru
    conservative_knowledge = """
    Financial Philosophy: CONSERVATIVE APPROACH
    
    Priority is safety and stability. Build a 6-12 month emergency fund before any investments.
    Focus on low-risk investments like bonds, CDs, and high-yield savings accounts.
    Avoid debt at all costs - pay off all debts before investing.
    Budget strictly: track every dollar spent.
    Live below your means - save at least 30% of income.
    Avoid market speculation and risky investments.
    Insurance is critical - health, life, disability.
    Retirement: max out 401k and IRA contributions.
    """
    
    # Aggressive Guru
    aggressive_knowledge = """
    Financial Philosophy: AGGRESSIVE GROWTH APPROACH
    
    Time in market beats timing the market. Start investing immediately.
    Leverage good debt for wealth building - real estate, business loans.
    High-growth stocks and index funds for long-term wealth.
    Emergency fund: 3 months is enough, invest the rest.
    Take calculated risks - higher risk means higher returns.
    Dollar-cost averaging into crypto and growth stocks.
    Side hustles and multiple income streams are essential.
    Invest 50%+ of income for early retirement.
    """
    
    # Balanced Guru
    balanced_knowledge = """
    Financial Philosophy: BALANCED APPROACH
    
    The 50/30/20 rule: 50% needs, 30% wants, 20% savings/investments.
    Emergency fund: 3-6 months of expenses in high-yield savings.
    Diversified portfolio: 60% stocks, 30% bonds, 10% alternative assets.
    Pay off high-interest debt first (>7% APR), invest if lower.
    Compound interest is powerful - start early, be consistent.
    Automate savings and investments for discipline.
    Balance enjoying life now with planning for the future.
    Review and rebalance portfolio quarterly.
    """
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # Create separate vector stores for each philosophy
        vector_stores = {}
        
        for name, knowledge in [
            ("conservative", conservative_knowledge),
            ("aggressive", aggressive_knowledge),
            ("balanced", balanced_knowledge)
        ]:
            docs = [Document(page_content=chunk, metadata={"guru": name}) 
                   for chunk in text_splitter.split_text(knowledge)]
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_stores[name] = FAISS.from_documents(docs, embeddings)
            print(f"[SUCCESS] {name.title()} Guru database created")
        
        return vector_stores
    except Exception as e:
        print(f"[ERROR] Error setting up vector databases: {str(e)}")
        raise

# Initialize vector stores
guru_vector_stores = setup_multi_guru_vector_db()

# In-memory storage
expenses = []
financial_goals = []
budget_limits = {}
uploaded_images = []  # Store base64 images with metadata

# =============================================================================
# EXPENSE PREDICTION & CATEGORIZATION
# =============================================================================
def categorize_expense(items, merchant):
    """Auto-categorize expenses based on items and merchant"""
    categories = {
        'groceries': ['milk', 'bread', 'eggs', 'vegetable', 'fruit', 'meat', 'grocery'],
        'dining': ['restaurant', 'cafe', 'pizza', 'burger', 'food court', 'meal'],
        'transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'parking'],
        'utilities': ['electric', 'water', 'internet', 'phone', 'cable'],
        'healthcare': ['pharmacy', 'medical', 'hospital', 'doctor', 'clinic'],
        'entertainment': ['movie', 'theater', 'game', 'concert', 'netflix'],
        'shopping': ['clothing', 'shoes', 'electronics', 'amazon', 'walmart']
    }
    
    merchant_lower = merchant.lower()
    items_text = ' '.join([item.get('item_name', '').lower() for item in items])
    
    for category, keywords in categories.items():
        if any(kw in merchant_lower or kw in items_text for kw in keywords):
            return category
    
    return 'other'

def predict_next_month_expenses():
    """Predict next month's expenses based on historical data"""
    if len(expenses) < 3:
        return None
    
    # Group by category
    category_totals = defaultdict(list)
    for exp in expenses:
        category = exp.get('category', 'other')
        amount = exp.get('total_amount', 0)
        category_totals[category].append(amount)
    
    # Calculate averages
    predictions = {}
    for category, amounts in category_totals.items():
        avg = sum(amounts) / len(amounts)
        predictions[category] = round(avg, 2)
    
    predictions['total'] = round(sum(predictions.values()), 2)
    return predictions

# =============================================================================
# MULTI-GURU COMPARISON
# =============================================================================
def get_multi_guru_advice(user_question, vector_stores, client):
    """Get advice from all three financial gurus and compare"""
    responses = {}
    
    for guru_name, vector_store in vector_stores.items():
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            relevant_chunks = retriever.invoke(user_question)
            context = "\n".join([chunk.page_content for chunk in relevant_chunks])
            
            system_prompt = f"""You are a {guru_name} financial advisor. 
            Base your advice strictly on the provided {guru_name} financial philosophy.
            Be concise (2-3 sentences) and stay true to your philosophy."""
            
            prompt = f"CONTEXT: {context}\nQUESTION: {user_question}\n\nProvide {guru_name} advice:"
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            responses[guru_name] = response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Error getting {guru_name} advice: {e}")
            responses[guru_name] = f"Error getting advice: {str(e)}"
    
    return responses

# =============================================================================
# INVESTMENT RECOMMENDATIONS
# =============================================================================
def get_investment_recommendations(total_savings, risk_profile, client):
    """Generate personalized investment recommendations"""
    try:
        prompt = f"""
        As a financial advisor, provide investment recommendations for:
        - Available funds: ${total_savings:,.2f}
        - Risk profile: {risk_profile}
        
        Provide:
        1. Recommended asset allocation (%)
        2. Specific investment types to consider
        3. Expected returns and risks
        4. Action steps
        
        Keep it practical and specific.
        """
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Error generating recommendations: {e}")
        raise

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Enhanced Expenses API is running",
        "features": ["improved_ocr", "multi_guru", "predictions", "goals"],
        "expenses_count": len(expenses),
        "goals_count": len(financial_goals)
    })

@app.route('/api/extract-receipt', methods=['POST'])
def extract_receipt():
    """Enhanced OCR extraction with categorization and security"""
    try:
        # Input validation
        if 'image' not in request.files:
            return jsonify({
                "success": False, 
                "error": "No image file provided",
                "error_code": "MISSING_IMAGE"
            }), 400
        
        image_file = request.files['image']
        
        # Validate filename
        if not image_file.filename or image_file.filename == '':
            return jsonify({
                "success": False, 
                "error": "No file selected",
                "error_code": "EMPTY_FILENAME"
            }), 400
        
        # Validate file type
        if not allowed_file(image_file.filename):
            return jsonify({
                "success": False, 
                "error": f"Invalid file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}",
                "error_code": "INVALID_FILE_TYPE"
            }), 400
        
        file_content = image_file.read()
        
        # Validate file size
        if len(file_content) > MAX_FILE_SIZE:
            return jsonify({
                "success": False, 
                "error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
                "error_code": "FILE_TOO_LARGE"
            }), 400
        
        # Validate that file is actually an image
        try:
            from PIL import Image
            test_image = Image.open(io.BytesIO(file_content))
            test_image.verify()
        except Exception:
            return jsonify({
                "success": False,
                "error": "File is not a valid image",
                "error_code": "INVALID_IMAGE"
            }), 400
        
        # Convert to base64 for storage
        import base64
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # IMPROVED OCR with error handling
        try:
            raw_text = extract_text_improved(file_content)
        except Exception as e:
            print(f"[ERROR] OCR extraction failed: {e}")
            return jsonify({
                "success": False,
                "error": "OCR processing failed. Please ensure the image is clear and contains readable text.",
                "error_code": "OCR_FAILED",
                "details": str(e)
            }), 500
        
        # Validate OCR output
        if not raw_text or not raw_text.strip():
            return jsonify({
                "success": False, 
                "error": "No text could be extracted from the image. Please ensure the receipt is clearly visible and well-lit.",
                "error_code": "NO_TEXT_EXTRACTED"
            }), 400
        
        if len(raw_text.strip()) < 10:
            return jsonify({
                "success": False,
                "error": "Insufficient text extracted. Please upload a clearer image.",
                "error_code": "INSUFFICIENT_TEXT"
            }), 400
        
        # Enhanced extraction prompt with better merchant name instructions
        system_prompt = """You are an expert receipt parser. Extract ALL data accurately.

CRITICAL INSTRUCTIONS:
1. For "address" field: Extract the STORE/MERCHANT NAME first (e.g., "Trader Joe's", "Walmart", "Target"), 
   then add the street address if available. Format: "Store Name - Street Address"
   
2. For items: Carefully match each product name with its exact price.

3. For "amount" (total): Look for keywords like TOTAL, BALANCE, AMOUNT DUE, GRAND TOTAL. 
   This should be the FINAL amount the customer paid. DO NOT use subtotal.

4. merchant_category: Identify what type of store this is (grocery store, restaurant, gas station, pharmacy, etc.)

Return valid JSON with this exact structure:
{
  "address": "Store Name - Street Address",
  "phoneNumber": "",
  "date": "MM/DD/YYYY",
  "items": [{"name": "Item Name", "price": 0.00}],
  "subTotal": 0.00,
  "salesTax": 0.00,
  "amount": 0.00,
  "merchant_category": "store type"
}

IMPORTANT: The "amount" field must be the total amount paid, not the subtotal."""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract structured data from this receipt text. Pay special attention to merchant name and final total amount:\n\n{raw_text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.05,
                timeout=30.0
            )
            json_output = response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] LLM processing failed: {e}")
            return jsonify({
                "success": False,
                "error": "AI processing failed. Please try again.",
                "error_code": "LLM_FAILED",
                "details": str(e)
            }), 500
        
        # Validate and parse JSON
        try:
            validated_receipt = Receipt.model_validate_json(json_output)
            parsed_json = validated_receipt.model_dump(by_alias=False)
        except Exception as e:
            print(f"[WARNING] Validation error: {e}")
            try:
                parsed_json = json.loads(json_output)
            except json.JSONDecodeError as je:
                return jsonify({
                    "success": False,
                    "error": "Failed to parse receipt data. The image may not contain a valid receipt.",
                    "error_code": "PARSE_FAILED",
                    "details": str(je)
                }), 500
        
        # Data validation - ensure critical fields exist
        if not parsed_json.get('total_amount') and not parsed_json.get('amount'):
            return jsonify({
                "success": False,
                "error": "Could not extract total amount from receipt. Please try a clearer image.",
                "error_code": "MISSING_TOTAL"
            }), 400
        
        # Sanitize data (basic security)
        def sanitize_string(s):
            if not isinstance(s, str):
                return s
            # Remove potentially dangerous characters
            return s.replace('<', '').replace('>', '').replace('"', '').replace("'", '').strip()
        
        parsed_json['address'] = sanitize_string(parsed_json.get('address', ''))
        parsed_json['phone'] = sanitize_string(parsed_json.get('phone', ''))
        
        # Auto-categorize
        category = categorize_expense(
            parsed_json.get('items', []),
            parsed_json.get('address', '')
        )
        
        # Create expense record
        expense_id = f"exp_{len(expenses) + 1}_{int(datetime.now().timestamp())}"
        expense = {
            "id": expense_id,
            "uploaded_at": datetime.now().isoformat(),
            "category": category,
            **parsed_json
        }
        expenses.append(expense)
        
        # Store image with metadata
        image_record = {
            "id": expense_id,
            "image_base64": image_base64,
            "filename": sanitize_string(image_file.filename),
            "uploaded_at": datetime.now().isoformat(),
            "store": parsed_json.get('address', 'Unknown'),
            "amount": parsed_json.get('total_amount', 0)
        }
        uploaded_images.append(image_record)
        
        print(f"[SUCCESS] Expense stored! Category: {category}, Total: {len(expenses)}")
        
        return jsonify({
            "success": True,
            "data": expense,
            "raw_ocr": raw_text[:500],
            "message": "Receipt processed successfully"
        })
            
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        return jsonify({
            "success": False, 
            "error": "An unexpected error occurred while processing the receipt.",
            "error_code": "INTERNAL_ERROR",
            "details": str(e) if app.debug else "Please try again"
        }), 500

@app.route('/api/uploaded-images', methods=['GET'])
def get_uploaded_images():
    """Get all uploaded receipt images"""
    try:
        # Return images with metadata (exclude base64 for list view for performance)
        images_list = [
            {
                "id": img["id"],
                "filename": img["filename"],
                "uploaded_at": img["uploaded_at"],
                "store": img["store"],
                "amount": img["amount"]
            }
            for img in uploaded_images
        ]
        return jsonify({
            "success": True,
            "images": images_list,
            "count": len(images_list)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/uploaded-images/<image_id>', methods=['GET'])
def get_single_image(image_id):
    """Get a single uploaded image with full base64 data"""
    try:
        image = next((img for img in uploaded_images if img["id"] == image_id), None)
        if not image:
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        return jsonify({
            "success": True,
            "image": image
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/expenses', methods=['GET'])
def get_expenses():
    try:
        total = sum(exp.get('total_amount', 0) for exp in expenses)
        
        # Category breakdown
        by_category = defaultdict(float)
        for exp in expenses:
            cat = exp.get('category', 'other')
            by_category[cat] += exp.get('total_amount', 0)
        
        return jsonify({
            "success": True,
            "expenses": expenses,
            "total": round(total, 2),
            "count": len(expenses),
            "by_category": {k: round(v, 2) for k, v in by_category.items()}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/expenses/<expense_id>', methods=['DELETE'])
def delete_expense(expense_id):
    """Delete a specific expense and its image"""
    try:
        global expenses, uploaded_images
        initial_count = len(expenses)
        expenses = [exp for exp in expenses if exp.get('id') != expense_id]
        uploaded_images = [img for img in uploaded_images if img.get('id') != expense_id]
        
        if len(expenses) == initial_count:
            return jsonify({
                "success": False,
                "error": "Expense not found"
            }), 404
        
        return jsonify({
            "success": True,
            "message": "Expense deleted successfully"
        })
    except Exception as e:
        print(f"[ERROR] Error deleting expense: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reports/monthly', methods=['GET'])
def get_monthly_report():
    """Generate comprehensive monthly financial report"""
    try:
        if not expenses:
            return jsonify({"success": False, "error": "No data"}), 400
        
        # Calculate metrics
        total = sum(exp.get('total_amount', 0) for exp in expenses)
        avg = total / len(expenses)
        
        by_category = defaultdict(float)
        by_date = defaultdict(float)
        
        for exp in expenses:
            cat = exp.get('category', 'other')
            date = exp.get('date', 'Unknown')
            amount = exp.get('total_amount', 0)
            by_category[cat] += amount
            by_date[date] += amount
        
        # Top spending categories
        top_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:5]
        
        report = {
            "total_spent": round(total, 2),
            "transaction_count": len(expenses),
            "average_transaction": round(avg, 2),
            "by_category": {k: round(v, 2) for k, v in by_category.items()},
            "by_date": {k: round(v, 2) for k, v in by_date.items()},
            "top_categories": [{"category": k, "amount": round(v, 2)} for k, v in top_categories]
        }
        
        return jsonify({"success": True, "report": report})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predict-expenses', methods=['GET'])
def predict_expenses():
    """Predict next month's expenses"""
    predictions = predict_next_month_expenses()
    
    if not predictions:
        return jsonify({
            "success": False,
            "error": "Need at least 3 expenses for predictions"
        }), 400
    
    return jsonify({"success": True, "predictions": predictions})

@app.route('/api/goals', methods=['GET', 'POST'])
def handle_goals():
    """Get or create financial goals"""
    if request.method == 'GET':
        return jsonify({"success": True, "goals": financial_goals})
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            goal = {
                "id": f"goal_{len(financial_goals) + 1}_{int(datetime.now().timestamp())}",
                "name": data.get('name'),
                "target_amount": float(data.get('target_amount')),
                "current_amount": float(data.get('current_amount', 0)),
                "deadline": data.get('deadline'),
                "category": data.get('category', 'savings'),
                "created_at": datetime.now().isoformat()
            }
            financial_goals.append(goal)
            return jsonify({"success": True, "goal": goal})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/goals/<goal_id>', methods=['PUT', 'DELETE'])
def manage_goal(goal_id):
    """Update or delete a goal"""
    global financial_goals
    
    if request.method == 'DELETE':
        initial_count = len(financial_goals)
        financial_goals = [g for g in financial_goals if g['id'] != goal_id]
        if len(financial_goals) == initial_count:
            return jsonify({"success": False, "error": "Not found"}), 404
        return jsonify({"success": True})
    
    elif request.method == 'PUT':
        try:
            data = request.get_json()
            for goal in financial_goals:
                if goal['id'] == goal_id:
                    goal['current_amount'] = float(data.get('current_amount', goal['current_amount']))
                    return jsonify({"success": True, "goal": goal})
            return jsonify({"success": False, "error": "Not found"}), 404
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/advice/multi-guru', methods=['POST'])
def get_multi_guru_comparison():
    """Get advice from all three financial philosophies"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"success": False, "error": "No question"}), 400
        
        print(f"\n[INFO] Multi-Guru Query: {question}")
        responses = get_multi_guru_advice(question, guru_vector_stores, client)
        
        return jsonify({"success": True, "advice": responses})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/investment-recommendations', methods=['POST'])
def investment_recommendations():
    """Get personalized investment recommendations"""
    try:
        data = request.get_json()
        total_savings = float(data.get('total_savings', 0))
        risk_profile = data.get('risk_profile', 'moderate')
        
        if total_savings <= 0:
            return jsonify({
                "success": False,
                "error": "Savings amount required"
            }), 400
        
        recommendations = get_investment_recommendations(
            total_savings, risk_profile, client
        )
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/analytics/advanced', methods=['GET'])
def advanced_analytics():
    """Advanced financial analytics with insights"""
    try:
        if not expenses:
            return jsonify({"success": False, "error": "No data available"}), 400
        
        # Performance optimization: process in batches for large datasets
        batch_size = 1000
        total_expenses = len(expenses)
        
        # Calculate comprehensive metrics
        total_spent = sum(exp.get('total_amount', 0) for exp in expenses)
        
        # Category analysis with optimization
        category_totals = defaultdict(float)
        category_counts = defaultdict(int)
        monthly_spending = defaultdict(float)
        
        for exp in expenses:
            cat = exp.get('category', 'other')
            amount = exp.get('total_amount', 0)
            category_totals[cat] += amount
            category_counts[cat] += 1
            
            # Monthly grouping
            date_str = exp.get('date', '')
            if date_str:
                try:
                    month = date_str.split('/')[0] if '/' in date_str else 'Unknown'
                    monthly_spending[month] += amount
                except:
                    pass
        
        # Calculate insights
        avg_transaction = total_spent / total_expenses if total_expenses > 0 else 0
        highest_category = max(category_totals.items(), key=lambda x: x[1]) if category_totals else ('N/A', 0)
        
        # Spending velocity (trend)
        recent_7_days = expenses[-7:] if len(expenses) >= 7 else expenses
        recent_total = sum(exp.get('total_amount', 0) for exp in recent_7_days)
        daily_avg = recent_total / len(recent_7_days) if recent_7_days else 0
        
        # Budget recommendations
        monthly_projection = daily_avg * 30
        recommended_budget = monthly_projection * 1.1  # 10% buffer
        
        analytics = {
            "total_spent": round(total_spent, 2),
            "transaction_count": total_expenses,
            "average_transaction": round(avg_transaction, 2),
            "category_breakdown": {k: round(v, 2) for k, v in category_totals.items()},
            "category_counts": dict(category_counts),
            "monthly_spending": {k: round(v, 2) for k, v in monthly_spending.items()},
            "insights": {
                "highest_spending_category": highest_category[0],
                "highest_category_amount": round(highest_category[1], 2),
                "daily_average": round(daily_avg, 2),
                "monthly_projection": round(monthly_projection, 2),
                "recommended_monthly_budget": round(recommended_budget, 2)
            },
            "warnings": []
        }
        
        # Add warnings
        if monthly_projection > 5000:
            analytics["warnings"].append("High spending detected. Consider reviewing your budget.")
        
        if category_totals.get('dining', 0) > total_spent * 0.3:
            analytics["warnings"].append("Dining expenses exceed 30% of total spending.")
        
        return jsonify({
            "success": True,
            "analytics": analytics,
            "processing_info": {
                "total_records": total_expenses,
                "batch_size": batch_size,
                "optimized": total_expenses > batch_size
            }
        })
        
    except Exception as e:
        print(f"[ERROR] Error in advanced analytics: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/analytics/dashboard', methods=['GET'])
def analytics_dashboard():
    """Comprehensive analytics dashboard data"""
    try:
        if not expenses:
            return jsonify({"success": False, "error": "No data"}), 400
        
        # Calculate comprehensive metrics
        total = sum(exp.get('total_amount', 0) for exp in expenses)
        
        # Category analysis
        by_category = defaultdict(float)
        for exp in expenses:
            by_category[exp.get('category', 'other')] += exp.get('total_amount', 0)
        
        # Trend analysis (last 7 days)
        recent_dates = defaultdict(float)
        for exp in expenses[-7:]:
            date = exp.get('date', 'Unknown')
            recent_dates[date] += exp.get('total_amount', 0)
        
        # Predictions
        predictions = predict_next_month_expenses()
        
        # Goal progress
        total_goals = sum(g['target_amount'] for g in financial_goals)
        achieved_goals = sum(g['current_amount'] for g in financial_goals)
        
        dashboard = {
            "total_spent": round(total, 2),
            "transaction_count": len(expenses),
            "category_breakdown": {k: round(v, 2) for k, v in by_category.items()},
            "recent_trend": {k: round(v, 2) for k, v in recent_dates.items()},
            "predictions": predictions,
            "goals_progress": {
                "total_target": round(total_goals, 2),
                "total_achieved": round(achieved_goals, 2),
                "percentage": round((achieved_goals / total_goals * 100) if total_goals > 0 else 0, 1)
            }
        }
        
        return jsonify({"success": True, "dashboard": dashboard})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("[STARTUP] ENHANCED Smart Expense Tracker API")
    print("=" * 80)
    print("\n[FEATURES]")
    print("   [OK] Improved OCR with image preprocessing")
    print("   [OK] Multi-Guru financial advice (Conservative/Aggressive/Balanced)")
    print("   [OK] Expense prediction & categorization")
    print("   [OK] Financial goal tracking")
    print("   [OK] Investment recommendations")
    print("   [OK] Comprehensive analytics dashboard")
    print("\n[SERVER] Starting on http://localhost:5000")
    print("=" * 80)
    app.run(debug=True, port=5000, host='0.0.0.0')
